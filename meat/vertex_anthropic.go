package meat

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"
)

const vertexAnthropicProvider = "vertex-anthropic"
const googleCloudScope = "https://www.googleapis.com/auth/cloud-platform"

// VertexAnthropicModel is a Model backed by Claude from Vertex AI Model Garden.
type VertexAnthropicModel struct {
	Model    string
	generate func(context.Context, anthropic.MessageNewParams) (*anthropic.Message, error)
}

// NewVertexAnthropicFromEnv builds a Claude-on-Vertex client using Application
// Default Credentials and meat-specific Google Cloud project/location variables.
func NewVertexAnthropicFromEnv(ctx context.Context, model string) (*VertexAnthropicModel, error) {
	location := vertexAnthropicLocation(model)
	project, err := googleProject(ctx)
	if err != nil {
		return nil, err
	}
	credentials, err := googleCredentials(ctx)
	if err != nil {
		return nil, fmt.Errorf("load Google credentials: %w", err)
	}
	client := anthropic.NewClient(
		option.WithoutEnvironmentDefaults(),
		// Four retries plus the initial request, with the SDK's exponential
		// backoff and Retry-After handling for 429s and transient 5xxs.
		option.WithMaxRetries(4),
		vertex.WithCredentials(ctx, location, project, credentials),
	)
	return &VertexAnthropicModel{
		Model: model,
		generate: func(ctx context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error) {
			return client.Messages.New(ctx, params)
		},
	}, nil
}

// vertexAnthropicLocation maps models that only support multi-region endpoints
// to the matching EU/US pool. Other models use the configured location directly.
func vertexAnthropicLocation(model string) string {
	location := os.Getenv("MEAT_GOOGLE_CLOUD_LOCATION")
	if location == "" {
		location = os.Getenv("MEAT_GOOGLE_CLOUD_REGION")
	}
	if !requiresVertexAnthropicMultiRegion(model) {
		if location == "" {
			return "global"
		}
		return location
	}

	normalized := strings.ToLower(strings.TrimSpace(location))
	switch {
	case normalized == "eu", normalized == "us", normalized == "global":
		return normalized
	case strings.HasPrefix(normalized, "europe-"):
		return "eu"
	case strings.HasPrefix(normalized, "us-"):
		return "us"
	default:
		return "eu"
	}
}

func requiresVertexAnthropicMultiRegion(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	model = strings.SplitN(model, "@", 2)[0]
	return model == "claude-opus-5" || model == "claude-sonnet-5"
}

// Generate implements Model using Anthropic's Vertex AI adapter.
func (m *VertexAnthropicModel) Generate(ctx context.Context, system string, messages []Message, tools []Tool) (*Response, error) {
	if m.Model == "" {
		return nil, fmt.Errorf("meat: VertexAnthropicModel.Model is empty")
	}
	if m.generate == nil {
		return nil, fmt.Errorf("meat: VertexAnthropicModel has no client")
	}
	anthropicMessages, err := toVertexAnthropicMessages(messages)
	if err != nil {
		return nil, err
	}
	anthropicTools, err := toVertexAnthropicTools(tools)
	if err != nil {
		return nil, err
	}
	resp, err := m.generate(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(m.Model),
		MaxTokens: int64(maxOutputTokens),
		System:    []anthropic.TextBlockParam{{Text: system}},
		Messages:  anthropicMessages,
		Tools:     anthropicTools,
	})
	if err != nil {
		return nil, fmt.Errorf("Anthropic Vertex generate: %w", err)
	}
	return responseFromVertexAnthropic(resp)
}

func toVertexAnthropicTools(tools []Tool) ([]anthropic.ToolUnionParam, error) {
	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		var schema anthropic.ToolInputSchemaParam
		if err := json.Unmarshal(tool.InputSchema, &schema); err != nil {
			return nil, fmt.Errorf("decode Anthropic Vertex tool schema for %s: %w", tool.Name, err)
		}
		param := anthropic.ToolUnionParamOfTool(schema, tool.Name)
		if tool.Description != "" {
			param.OfTool.Description = anthropic.String(tool.Description)
		}
		out = append(out, param)
	}
	return out, nil
}

func toVertexAnthropicMessages(messages []Message) ([]anthropic.MessageParam, error) {
	out := make([]anthropic.MessageParam, 0, len(messages))
	for _, message := range messages {
		blocks := make([]anthropic.ContentBlockParamUnion, 0, len(message.Content))
		for _, block := range message.Content {
			if block.Provider == vertexAnthropicProvider && len(block.ProviderData) > 0 {
				var providerBlock anthropic.ContentBlockParamUnion
				if err := json.Unmarshal(block.ProviderData, &providerBlock); err != nil {
					return nil, fmt.Errorf("decode Anthropic Vertex provider state: %w", err)
				}
				blocks = append(blocks, providerBlock)
				continue
			}
			switch block.Type {
			case "text":
				blocks = append(blocks, anthropic.NewTextBlock(block.Text))
			case "tool_use":
				var input any
				if len(block.ToolInput) > 0 {
					if err := json.Unmarshal(block.ToolInput, &input); err != nil {
						return nil, fmt.Errorf("decode Anthropic Vertex tool input for %s: %w", block.ToolName, err)
					}
				}
				if input == nil {
					input = map[string]any{}
				}
				blocks = append(blocks, anthropic.NewToolUseBlock(block.ID, input, block.ToolName))
			case "tool_result":
				blocks = append(blocks, anthropic.NewToolResultBlock(block.ToolUseID, block.ToolResult, block.ToolError))
			case "provider_state":
				return nil, fmt.Errorf("Anthropic Vertex provider state is missing opaque response data")
			default:
				return nil, fmt.Errorf("Anthropic Vertex: unsupported message block type %q", block.Type)
			}
		}
		role := anthropic.MessageParamRoleUser
		if message.Role == RoleAssistant {
			role = anthropic.MessageParamRoleAssistant
		}
		out = append(out, anthropic.MessageParam{Role: role, Content: blocks})
	}
	return out, nil
}

func responseFromVertexAnthropic(resp *anthropic.Message) (*Response, error) {
	if resp == nil {
		return nil, fmt.Errorf("Anthropic Vertex returned an empty response")
	}
	if resp.StopReason == anthropic.StopReasonMaxTokens {
		return nil, fmt.Errorf("Anthropic Vertex response truncated at max_tokens (%d)", maxOutputTokens)
	}
	out := &Response{
		InputTokens:  int(resp.Usage.InputTokens),
		OutputTokens: int(resp.Usage.OutputTokens),
	}
	for _, content := range resp.Content {
		raw := []byte(content.RawJSON())
		if len(raw) == 0 {
			var err error
			raw, err = json.Marshal(content)
			if err != nil {
				return nil, fmt.Errorf("encode Anthropic Vertex provider state: %w", err)
			}
		}
		state := json.RawMessage(raw)
		switch content.Type {
		case "text":
			out.Content = append(out.Content, Block{
				Type: "text", Text: content.Text,
				Provider: vertexAnthropicProvider, ProviderData: state,
			})
		case "tool_use":
			out.Content = append(out.Content, Block{
				Type: "tool_use", ID: content.ID, ToolName: content.Name, ToolInput: content.Input,
				Provider: vertexAnthropicProvider, ProviderData: state,
			})
		default:
			out.Content = append(out.Content, Block{
				Type: "provider_state", Provider: vertexAnthropicProvider, ProviderData: state,
			})
		}
	}
	if len(out.Content) == 0 {
		return nil, fmt.Errorf("Anthropic Vertex returned no usable content")
	}
	return out, nil
}
