package meat

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"google.golang.org/genai"
)

const geminiProvider = "gemini"

// GeminiModel is a Model backed by Gemini Enterprise Agent Platform. The SDK
// uses Application Default Credentials; model names are passed through unchanged.
type GeminiModel struct {
	Model    string
	generate func(context.Context, string, []*genai.Content, *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error)
}

// NewGeminiFromEnv builds a Gemini Enterprise model from Google Cloud's
// standard project, location, and Application Default Credentials configuration.
func NewGeminiFromEnv(ctx context.Context, model string) (*GeminiModel, error) {
	project := os.Getenv("GOOGLE_CLOUD_PROJECT")
	if project == "" {
		return nil, fmt.Errorf("no Google Cloud project: set GOOGLE_CLOUD_PROJECT")
	}
	location := geminiLocation(model)
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend:  genai.BackendEnterprise,
		Project:  project,
		Location: location,
	})
	if err != nil {
		return nil, fmt.Errorf("create Gemini Enterprise client: %w", err)
	}
	return &GeminiModel{
		Model: model,
		generate: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
			return client.Models.GenerateContent(ctx, model, contents, config)
		},
	}, nil
}

// geminiLocation keeps regional models on the configured location and routes
// models that Agent Platform only publishes globally to the global endpoint.
func geminiLocation(model string) string {
	if requiresGeminiGlobal(model) {
		return "global"
	}
	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		location = os.Getenv("GOOGLE_CLOUD_REGION")
	}
	if location == "" {
		return "global"
	}
	return location
}

func requiresGeminiGlobal(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	return model == "gemini-3.6-flash" || strings.HasPrefix(model, "gemini-3.6-flash-")
}

// Generate implements Model using Gemini Enterprise's GenerateContent API.
func (m *GeminiModel) Generate(ctx context.Context, system string, messages []Message, tools []Tool) (*Response, error) {
	if m.Model == "" {
		return nil, fmt.Errorf("meat: GeminiModel.Model is empty")
	}
	if m.generate == nil {
		return nil, fmt.Errorf("meat: GeminiModel has no client")
	}
	contents, err := toGeminiContents(messages)
	if err != nil {
		return nil, err
	}
	config := &genai.GenerateContentConfig{
		SystemInstruction: &genai.Content{Parts: []*genai.Part{genai.NewPartFromText(system)}},
		MaxOutputTokens:   int32(maxOutputTokens),
		Tools:             toGeminiTools(tools),
	}
	resp, err := m.generate(ctx, m.Model, contents, config)
	if err != nil {
		return nil, fmt.Errorf("Gemini Enterprise generate: %w", err)
	}
	return responseFromGemini(resp)
}

func toGeminiTools(tools []Tool) []*genai.Tool {
	if len(tools) == 0 {
		return nil
	}
	declarations := make([]*genai.FunctionDeclaration, 0, len(tools))
	for _, tool := range tools {
		var schema any
		if err := json.Unmarshal(tool.InputSchema, &schema); err != nil {
			// Tool schemas are validated when meat constructs them. Preserve the raw
			// value here; GenerateContent will surface any unexpected SDK error.
			schema = map[string]any{"type": "object"}
		}
		declarations = append(declarations, &genai.FunctionDeclaration{
			Name:                 tool.Name,
			Description:          tool.Description,
			ParametersJsonSchema: schema,
		})
	}
	return []*genai.Tool{{FunctionDeclarations: declarations}}
}

type geminiCall struct {
	name       string
	providerID string
}

func toGeminiContents(messages []Message) ([]*genai.Content, error) {
	contents := make([]*genai.Content, 0, len(messages))
	calls := make(map[string]geminiCall)
	for _, message := range messages {
		var role genai.Role = genai.RoleUser
		if message.Role == RoleAssistant {
			role = genai.RoleModel
		}
		content := genai.NewContentFromParts(nil, role)
		for _, block := range message.Content {
			if block.Provider == geminiProvider && len(block.ProviderData) > 0 {
				var part genai.Part
				if err := json.Unmarshal(block.ProviderData, &part); err != nil {
					return nil, fmt.Errorf("decode Gemini provider state: %w", err)
				}
				content.Parts = append(content.Parts, &part)
				if block.Type == "tool_use" && part.FunctionCall != nil {
					calls[block.ID] = geminiCall{name: part.FunctionCall.Name, providerID: part.FunctionCall.ID}
				}
				continue
			}

			switch block.Type {
			case "text":
				content.Parts = append(content.Parts, genai.NewPartFromText(block.Text))
			case "tool_use":
				var args map[string]any
				if len(block.ToolInput) > 0 {
					if err := json.Unmarshal(block.ToolInput, &args); err != nil {
						return nil, fmt.Errorf("decode Gemini tool input for %s: %w", block.ToolName, err)
					}
				}
				if args == nil {
					args = map[string]any{}
				}
				content.Parts = append(content.Parts, &genai.Part{FunctionCall: &genai.FunctionCall{
					ID: block.ID, Name: block.ToolName, Args: args,
				}})
				calls[block.ID] = geminiCall{name: block.ToolName, providerID: block.ID}
			case "tool_result":
				call, ok := calls[block.ToolUseID]
				if !ok {
					return nil, fmt.Errorf("Gemini tool result references unknown call %q", block.ToolUseID)
				}
				key := "output"
				if block.ToolError {
					key = "error"
				}
				content.Parts = append(content.Parts, &genai.Part{FunctionResponse: &genai.FunctionResponse{
					ID: call.providerID, Name: call.name, Response: map[string]any{key: block.ToolResult},
				}})
			case "provider_state":
				return nil, fmt.Errorf("Gemini provider state is missing opaque response data")
			default:
				return nil, fmt.Errorf("Gemini: unsupported message block type %q", block.Type)
			}
		}
		contents = append(contents, content)
	}
	return contents, nil
}

func responseFromGemini(resp *genai.GenerateContentResponse) (*Response, error) {
	if resp == nil {
		return nil, fmt.Errorf("Gemini returned an empty response")
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0] == nil {
		if resp.PromptFeedback != nil {
			return nil, fmt.Errorf("Gemini returned no candidates: %s", resp.PromptFeedback.BlockReason)
		}
		return nil, fmt.Errorf("Gemini returned no candidates")
	}
	candidate := resp.Candidates[0]
	if candidate.FinishReason == genai.FinishReasonMaxTokens {
		return nil, fmt.Errorf("Gemini response truncated at max output tokens (%d)", maxOutputTokens)
	}
	if candidate.FinishReason != "" && candidate.FinishReason != genai.FinishReasonStop {
		return nil, fmt.Errorf("Gemini response stopped with %s: %s", candidate.FinishReason, candidate.FinishMessage)
	}
	if candidate.Content == nil {
		return nil, fmt.Errorf("Gemini returned a candidate without content")
	}

	out := &Response{}
	if resp.UsageMetadata != nil {
		out.InputTokens = int(resp.UsageMetadata.PromptTokenCount)
		out.OutputTokens = int(resp.UsageMetadata.CandidatesTokenCount + resp.UsageMetadata.ThoughtsTokenCount)
	}
	for i, part := range candidate.Content.Parts {
		if part == nil {
			continue
		}
		raw, err := json.Marshal(part)
		if err != nil {
			return nil, fmt.Errorf("encode Gemini provider state: %w", err)
		}
		state := json.RawMessage(raw)
		switch {
		case part.FunctionCall != nil:
			input, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("encode Gemini tool input: %w", err)
			}
			id := part.FunctionCall.ID
			if id == "" {
				id = fmt.Sprintf("gemini-call-%d", i+1)
			}
			out.Content = append(out.Content, Block{
				Type: "tool_use", ID: id, ToolName: part.FunctionCall.Name, ToolInput: input,
				Provider: geminiProvider, ProviderData: state,
			})
		case part.Text != "" && !part.Thought:
			out.Content = append(out.Content, Block{
				Type: "text", Text: part.Text, Provider: geminiProvider, ProviderData: state,
			})
		default:
			// Thought signatures and unfamiliar parts must be replayed exactly on
			// the next turn even though the provider-agnostic agent ignores them.
			out.Content = append(out.Content, Block{
				Type: "provider_state", Provider: geminiProvider, ProviderData: state,
			})
		}
	}
	if len(out.Content) == 0 {
		return nil, fmt.Errorf("Gemini returned no usable content")
	}
	return out, nil
}

func isVertexModel(model string) bool { return strings.HasPrefix(model, "vertex/") }
