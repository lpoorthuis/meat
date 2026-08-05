package meat

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"google.golang.org/genai"
)

func TestGeminiGenerateAndReplayToolCall(t *testing.T) {
	var gotModel string
	var gotContents []*genai.Content
	var gotConfig *genai.GenerateContentConfig
	model := &GeminiModel{
		Model: "gemini-test",
		generate: func(_ context.Context, name string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
			gotModel, gotContents, gotConfig = name, contents, config
			return &genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{
					FinishReason: genai.FinishReasonStop,
					Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
						{Text: "internal", Thought: true, ThoughtSignature: []byte("signature")},
						{FunctionCall: &genai.FunctionCall{Name: "submit", Args: map[string]any{"summary": "done"}}},
					}},
				}},
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount: 7, CandidatesTokenCount: 3, ThoughtsTokenCount: 2,
				},
			}, nil
		},
	}

	resp, err := model.Generate(context.Background(), "system", []Message{{
		Role: RoleUser, Content: []Block{textBlock("diff")},
	}}, []Tool{{Name: "submit", Description: "finish", InputSchema: json.RawMessage(`{"type":"object","properties":{"summary":{"type":"string"}}}`)}})
	if err != nil {
		t.Fatal(err)
	}
	if gotModel != "gemini-test" || len(gotContents) != 1 || gotContents[0].Parts[0].Text != "diff" {
		t.Fatalf("unexpected request: model=%q contents=%+v", gotModel, gotContents)
	}
	if gotConfig.SystemInstruction.Role != "" || gotConfig.SystemInstruction.Parts[0].Text != "system" {
		t.Fatalf("unexpected system instruction: %+v", gotConfig.SystemInstruction)
	}
	if len(gotConfig.Tools) != 1 || gotConfig.Tools[0].FunctionDeclarations[0].Name != "submit" {
		t.Fatalf("unexpected tools: %+v", gotConfig.Tools)
	}
	if resp.InputTokens != 7 || resp.OutputTokens != 5 {
		t.Fatalf("unexpected usage: %+v", resp)
	}
	if len(resp.Content) != 2 || resp.Content[0].Type != "provider_state" || resp.Content[1].Type != "tool_use" {
		t.Fatalf("unexpected response blocks: %+v", resp.Content)
	}
	if resp.Content[1].ID != "gemini-call-2" || string(resp.Content[1].ToolInput) != `{"summary":"done"}` {
		t.Fatalf("unexpected tool call: %+v", resp.Content[1])
	}

	messages := []Message{
		{Role: RoleUser, Content: []Block{textBlock("diff")}},
		{Role: RoleAssistant, Content: resp.Content},
		{Role: RoleUser, Content: []Block{{Type: "tool_result", ToolUseID: resp.Content[1].ID, ToolResult: "ok"}}},
	}
	replayed, err := toGeminiContents(messages)
	if err != nil {
		t.Fatal(err)
	}
	if len(replayed) != 3 || string(replayed[1].Parts[0].ThoughtSignature) != "signature" {
		t.Fatalf("thought signature was not replayed: %+v", replayed)
	}
	result := replayed[2].Parts[0].FunctionResponse
	if result == nil || result.Name != "submit" || result.ID != "" || result.Response["output"] != "ok" {
		t.Fatalf("unexpected replayed tool result: %+v", result)
	}
}

func TestGeminiMaxTokensIsError(t *testing.T) {
	_, err := responseFromGemini(&genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
		FinishReason: genai.FinishReasonMaxTokens,
		Content:      &genai.Content{},
	}}})
	if err == nil || !strings.Contains(err.Error(), "truncated") {
		t.Fatalf("error = %v, want truncation error", err)
	}
}
