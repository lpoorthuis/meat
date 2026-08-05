package meat

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

func TestVertexAnthropicGenerateAndReplayToolCall(t *testing.T) {
	var got anthropic.MessageNewParams
	model := &VertexAnthropicModel{
		Model: "claude-test@001",
		generate: func(_ context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error) {
			got = params
			var response anthropic.Message
			if err := json.Unmarshal([]byte(`{
				"content":[{"type":"tool_use","id":"tool-1","name":"submit","input":{"summary":"done"}}],
				"stop_reason":"tool_use",
				"usage":{"input_tokens":11,"output_tokens":4}
			}`), &response); err != nil {
				t.Fatal(err)
			}
			return &response, nil
		},
	}

	resp, err := model.Generate(context.Background(), "system", []Message{{
		Role: RoleUser, Content: []Block{textBlock("diff")},
	}}, []Tool{{Name: "submit", Description: "finish", InputSchema: json.RawMessage(`{"type":"object","properties":{"summary":{"type":"string"}}}`)}})
	if err != nil {
		t.Fatal(err)
	}
	if got.Model != "claude-test@001" || got.System[0].Text != "system" || got.Messages[0].Content[0].OfText.Text != "diff" {
		t.Fatalf("unexpected request: %+v", got)
	}
	if len(got.Tools) != 1 || got.Tools[0].OfTool.Name != "submit" || got.Tools[0].GetDescription() == nil || *got.Tools[0].GetDescription() != "finish" {
		t.Fatalf("unexpected tools: %+v", got.Tools)
	}
	if resp.InputTokens != 11 || resp.OutputTokens != 4 || len(resp.Content) != 1 {
		t.Fatalf("unexpected response: %+v", resp)
	}
	call := resp.Content[0]
	if call.Type != "tool_use" || call.ID != "tool-1" || call.Provider != vertexAnthropicProvider {
		t.Fatalf("unexpected tool call: %+v", call)
	}

	messages := []Message{
		{Role: RoleAssistant, Content: resp.Content},
		{Role: RoleUser, Content: []Block{{Type: "tool_result", ToolUseID: call.ID, ToolResult: "ok"}}},
	}
	replayed, err := toVertexAnthropicMessages(messages)
	if err != nil {
		t.Fatal(err)
	}
	if replayed[0].Content[0].OfToolUse == nil || replayed[0].Content[0].OfToolUse.ID != "tool-1" {
		t.Fatalf("tool call provider state was not replayed: %+v", replayed[0])
	}
	if replayed[1].Content[0].OfToolResult == nil || replayed[1].Content[0].OfToolResult.ToolUseID != "tool-1" {
		t.Fatalf("tool result was not converted: %+v", replayed[1])
	}
}

func TestVertexAnthropicLocation(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		location string
		region   string
		want     string
	}{
		{name: "Opus 5 maps European region to EU", model: "claude-opus-5", location: "europe-west1", want: "eu"},
		{name: "Sonnet 5 maps US region to US", model: "claude-sonnet-5", location: "us-east5", want: "us"},
		{name: "Opus 5 defaults to EU", model: "claude-opus-5", want: "eu"},
		{name: "Opus 5 preserves global", model: "claude-opus-5", location: "global", want: "global"},
		{name: "Opus 5 snapshot is multi-region", model: "claude-opus-5@20260801", location: "europe-west1", want: "eu"},
		{name: "regional model preserves location", model: "claude-opus-4.7", location: "europe-west1", want: "europe-west1"},
		{name: "regional model reads region alias", model: "claude-opus-4.7", region: "us-east5", want: "us-east5"},
		{name: "regional model defaults global", model: "claude-opus-4.7", want: "global"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv("GOOGLE_CLOUD_LOCATION", test.location)
			t.Setenv("GOOGLE_CLOUD_REGION", test.region)
			if got := vertexAnthropicLocation(test.model); got != test.want {
				t.Errorf("vertexAnthropicLocation(%q) = %q, want %q", test.model, got, test.want)
			}
		})
	}
}

func TestVertexAnthropicMaxTokensIsError(t *testing.T) {
	_, err := responseFromVertexAnthropic(&anthropic.Message{StopReason: anthropic.StopReasonMaxTokens})
	if err == nil || !strings.Contains(err.Error(), "truncated") {
		t.Fatalf("error = %v, want truncation error", err)
	}
}
