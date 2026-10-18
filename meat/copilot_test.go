package meat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	copilot "github.com/github/copilot-sdk/go"
	"github.com/github/copilot-sdk/go/rpc"
)

type fakeCopilotSession struct {
	events       chan copilot.SessionEvent
	prompts      []string
	results      []Block
	requestIDs   []string
	closed       int
	onSend       func(string)
	onToolResult func(string, Block)
	sendErr      error
	resultErr    error
}

func newFakeCopilotSession() *fakeCopilotSession {
	return &fakeCopilotSession{events: make(chan copilot.SessionEvent, 64)}
}

func (s *fakeCopilotSession) Send(_ context.Context, prompt string) error {
	s.prompts = append(s.prompts, prompt)
	if s.onSend != nil {
		s.onSend(prompt)
	}
	return s.sendErr
}

func (s *fakeCopilotSession) ToolResult(_ context.Context, id string, block Block) error {
	s.results = append(s.results, block)
	s.requestIDs = append(s.requestIDs, id)
	if s.onToolResult != nil {
		s.onToolResult(id, block)
	}
	return s.resultErr
}

func (s *fakeCopilotSession) Events() <-chan copilot.SessionEvent { return s.events }
func (s *fakeCopilotSession) Close()                              { s.closed++ }

func (s *fakeCopilotSession) emit(data copilot.SessionEventData) {
	s.events <- copilot.SessionEvent{Data: data}
}

func (s *fakeCopilotSession) call(id, name string, args json.RawMessage) {
	var input any
	if err := json.Unmarshal(args, &input); err != nil {
		panic(err)
	}
	s.emit(&copilot.AssistantMessageData{ToolRequests: []rpc.AssistantMessageToolRequest{{ToolCallID: id, Name: name, Arguments: input}}})
	s.emit(&copilot.ExternalToolRequestedData{ToolCallID: id, ToolName: name, RequestID: "request-" + id})
}

func fakeCopilotModel(s *fakeCopilotSession) *CopilotModel {
	return &CopilotModel{Model: "gpt-6.1-sol", open: func(context.Context, copilotOptions) (copilotSession, error) { return s, nil }}
}

func copilotUser(text string) []Message {
	return []Message{{Role: RoleUser, Content: []Block{textBlock(text)}}}
}

func copilotNext(messages []Message, response *Response, blocks ...Block) []Message {
	return append(messages, Message{Role: RoleAssistant, Content: response.Content}, Message{Role: RoleUser, Content: blocks})
}

func TestCopilotExplicitModelRouting(t *testing.T) {
	t.Setenv("MEAT_COPILOT_CLI", "/custom/copilot")
	for _, id := range []string{
		"auto", "claude-haiku-4.5", "claude-haiku-5.5", "claude-sonnet-5", "claude-sonnet-5.5",
		"gpt-5-mini", "gpt-5.3-codex", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5",
		"gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-sol-fast", "gpt-5.6-terra",
		"gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol", "grok-4.5", "grok-4.6", "grok-4.7", "mai-code-1.1-flash",
	} {
		selector := "copilot/" + id
		t.Setenv("MEAT_MODEL", selector)
		if got := ResolveModel(""); got != selector {
			t.Fatalf("environment model changed: got %q, want %q", got, selector)
		}
		model, err := NewModelFromEnv(context.Background(), "")
		if err != nil {
			t.Fatal(err)
		}
		m, ok := model.(*CopilotModel)
		if !ok {
			t.Fatalf("%s -> %T", selector, model)
		}
		if m.Model != id || m.CLIPath != "/custom/copilot" || m.run != nil {
			t.Fatalf("unexpected lazy provider: model=%q path=%q run=%v", m.Model, m.CLIPath, m.run)
		}
	}
	// An explicit flag still takes precedence over the environment, unchanged.
	t.Setenv("MEAT_MODEL", "copilot/claude-haiku-5.5")
	model, err := NewModelFromEnv(context.Background(), "copilot/gpt-6.1-sol")
	if err != nil || model.(*CopilotModel).Model != "gpt-6.1-sol" {
		t.Fatalf("explicit model selection: model=%v error=%v", model, err)
	}
	for _, selector := range []string{"copilot", "copilot/", "copilot/ ", "copilot/gpt/other", "copilot/gpt\n"} {
		t.Setenv("MEAT_MODEL", selector)
		if ResolveModel("") != selector {
			t.Errorf("unexpected rewrite of %q", selector)
		}
		for _, argument := range []string{"", selector} {
			if _, err := NewModelFromEnv(context.Background(), argument); err == nil {
				t.Errorf("accepted invalid selector %q", selector)
			}
		}
	}
	t.Setenv("MEAT_COPILOT_CLI", "")
	t.Setenv("COPILOT_CLI_PATH", "/sdk/copilot")
	m, err := NewCopilotFromEnv("gpt-6.1-sol")
	if err != nil || m.CLIPath != "/sdk/copilot" {
		t.Fatalf("SDK CLI override: model=%v err=%v", m, err)
	}
}

func TestCopilotRequiresExplicitModel(t *testing.T) {
	// Neither the constructor nor a zero-value model may invent a model ID.
	t.Setenv("MEAT_MODEL", "copilot/gpt-6.1-sol")
	if _, err := NewCopilotFromEnv(""); err == nil {
		t.Fatal("constructor accepted an empty model")
	}
	m := &CopilotModel{open: func(context.Context, copilotOptions) (copilotSession, error) {
		t.Fatal("must reject an empty model before opening a session")
		return nil, nil
	}}
	_, err := m.Generate(context.Background(), "rubric", copilotUser("diff"), nil)
	if err == nil || !strings.Contains(err.Error(), "model id is empty") {
		t.Fatalf("error = %v, want an explicit model requirement", err)
	}
}

func TestCopilotConfigurationPreservesLoginAndRestrictsRuntime(t *testing.T) {
	options := copilotClientOptions("/bin/copilot", "/isolated")
	if options.BaseDirectory != "" || options.Mode != copilot.ModeCopilotCli || options.GitHubToken != "" || options.Env != nil {
		t.Fatal("must preserve the normal Copilot credential source, not replace it with an empty home/token")
	}
	connection, ok := options.Connection.(copilot.StdioConnection)
	if !ok || connection.Path != "/bin/copilot" || connection.Env != nil || options.WorkingDirectory != "/isolated" {
		t.Fatalf("unexpected connection: %#v", options)
	}
	for _, flag := range []string{"--no-auto-update", "--disable-builtin-mcps", "--no-custom-instructions"} {
		if !slices.Contains(connection.Args, flag) {
			t.Errorf("missing runtime restriction %s", flag)
		}
	}
	tools := (&toolbox{root: "/private/repo"}).tools()
	config, err := copilotSessionConfig(copilotOptions{Model: "gpt-6.1-sol", System: "exact rubric", Tools: tools}, "/isolated")
	if err != nil {
		t.Fatal(err)
	}
	if config.SystemMessage.Mode != "replace" || config.SystemMessage.Content != "exact rubric" || config.Provider != nil || config.ConfigDirectory != "/isolated" || config.WorkingDirectory != "/isolated" {
		t.Fatalf("unexpected session config: %#v", config)
	}
	for name, value := range map[string]*bool{
		"config discovery": config.EnableConfigDiscovery, "instruction discovery": config.EnableOnDemandInstructionDiscovery,
		"hooks": config.EnableFileHooks, "git context": config.EnableHostGitOperations,
		"session store": config.EnableSessionStore, "GitHub cache": config.EnableGitHubCacheStore,
		"skills": config.EnableSkills, "compaction": config.InfiniteSessions.Enabled,
	} {
		if value == nil || *value {
			t.Errorf("%s must be explicitly disabled", name)
		}
	}
	if config.SkipCustomInstructions == nil || !*config.SkipCustomInstructions || config.Memory.Enabled {
		t.Fatal("instructions and memory must be disabled")
	}
	for i, tool := range config.Tools {
		if tool.Handler != nil || !tool.SkipPermission || !tool.OverridesBuiltInTool || tool.Defer != copilot.ToolDeferNever {
			t.Errorf("tool %s must be declaration-only, preloaded, and owned by meat", tool.Name)
		}
		if config.AvailableTools[i] != "custom:"+tools[i].Name {
			t.Errorf("non-custom tool allowed: %v", config.AvailableTools)
		}
		var schema map[string]any
		if err := json.Unmarshal(tools[i].InputSchema, &schema); err != nil || !reflect.DeepEqual(tool.Parameters, schema) {
			t.Errorf("tool schema changed for %s", tool.Name)
		}
	}
	empty, err := copilotSessionConfig(copilotOptions{}, "/isolated")
	if err != nil || empty.AvailableTools == nil || len(empty.AvailableTools) != 0 {
		t.Fatal("no tools must mean an explicit empty allowlist, not runtime defaults")
	}
	for _, schema := range []string{`null`, `[]`, `broken`} {
		if _, err := copilotSessionConfig(copilotOptions{Tools: []Tool{{Name: "bad", InputSchema: json.RawMessage(schema)}}}, ""); err == nil {
			t.Errorf("accepted invalid schema %s", schema)
		}
	}
}

func TestCopilotGenerateToolRoundTrip(t *testing.T) {
	s := newFakeCopilotSession()
	m := fakeCopilotModel(s)
	defer m.Close()
	tools := (&toolbox{root: "/repo"}).tools()
	messages := copilotUser("numbered diff")
	in, out := int64(123), int64(17)
	s.onSend = func(prompt string) {
		if prompt != "numbered diff" {
			t.Fatalf("prompt = %q", prompt)
		}
		s.emit(&copilot.AssistantUsageData{InputTokens: &in, OutputTokens: &out})
		// Some providers split one response at reasoning boundaries.
		s.emit(&copilot.AssistantMessageData{Content: "Inspecting source."})
		// Dispatch can arrive before the assistant event is observed.
		s.emit(&copilot.ExternalToolRequestedData{ToolName: "read_file", ToolCallID: "read", RequestID: "sdk-read"})
		s.emit(&copilot.AssistantMessageData{ToolRequests: []rpc.AssistantMessageToolRequest{
			{Name: "read_file", ToolCallID: "read", Arguments: map[string]string{"path": "a.go"}},
			{Name: "preview_plan", ToolCallID: "preview", Arguments: map[string]any{"remove": []int{}, "replace": []int{}, "fold": []int{}}},
		}})
		s.emit(&copilot.ExternalToolRequestedData{ToolName: "preview_plan", ToolCallID: "preview", RequestID: "sdk-preview"})
	}
	response, err := m.Generate(context.Background(), "rubric", messages, tools)
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Content) != 3 || response.Content[1].ToolName != "read_file" || string(response.Content[1].ToolInput) != `{"path":"a.go"}` || response.InputTokens != 123 || response.OutputTokens != 17 {
		t.Fatalf("unexpected response: %+v", response)
	}
	s.onToolResult = func(_ string, _ Block) {
		if len(s.results) == 2 {
			s.emit(&copilot.AssistantUsageData{InputTokens: &in, OutputTokens: &out})
			s.emit(&copilot.AssistantMessageData{Content: "Done."})
			s.emit(&copilot.SessionIdleData{})
		}
	}
	// Return in a different order from the model's calls; one result is a
	// validation error, not a transport failure.
	messages = copilotNext(messages, response,
		Block{Type: "tool_result", ToolUseID: "preview", ToolResult: "invalid fold", ToolError: true},
		Block{Type: "tool_result", ToolUseID: "read", ToolResult: "file contents"},
	)
	response, err = m.Generate(context.Background(), "rubric", messages, tools)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(s.requestIDs, []string{"sdk-preview", "sdk-read"}) || !s.results[0].ToolError || len(s.prompts) != 1 {
		t.Fatalf("tool handoff changed: ids=%v results=%v prompts=%v", s.requestIDs, s.results, s.prompts)
	}
	if response.InputTokens != 123 || response.OutputTokens != 17 || response.Content[0].Text != "Done." {
		t.Fatalf("usage must be per response, not cumulative: %+v", response)
	}
}

func TestCopilotWaitsForAllResponseChunksAndIgnoresSubagents(t *testing.T) {
	s := newFakeCopilotSession()
	m := fakeCopilotModel(s)
	defer m.Close()
	first, last, count := int64(0), int64(1), int64(2)
	agent := "unrelated-subagent"
	s.events <- copilot.SessionEvent{AgentID: &agent, Data: &copilot.SessionErrorData{ErrorType: "quota", Message: "not our response"}}
	s.emit(&copilot.AssistantMessageData{ChunkIndex: &first, ChunkCount: &count, ToolRequests: []rpc.AssistantMessageToolRequest{{Name: "preview_plan", ToolCallID: "a"}}})
	s.emit(&copilot.ExternalToolRequestedData{ToolName: "preview_plan", ToolCallID: "a", RequestID: "request-a"})
	s.emit(&copilot.AssistantMessageData{ChunkIndex: &last, ChunkCount: &count, ToolRequests: []rpc.AssistantMessageToolRequest{{Name: "submit", ToolCallID: "b"}}})
	s.emit(&copilot.ExternalToolRequestedData{ToolName: "submit", ToolCallID: "b", RequestID: "request-b"})
	response, err := m.Generate(context.Background(), "rubric", copilotUser("diff"), (&toolbox{}).tools())
	if err != nil || len(response.Content) != 2 || len(m.run.pending) != 2 {
		t.Fatalf("partial response: response=%+v error=%v", response, err)
	}
}

func TestCopilotGenerateTextNudge(t *testing.T) {
	s := newFakeCopilotSession()
	m := fakeCopilotModel(s)
	defer m.Close()
	s.onSend = func(string) {
		s.emit(&copilot.AssistantMessageData{Content: "Working."})
		s.emit(&copilot.SessionIdleData{})
	}
	messages := copilotUser("diff")
	response, err := m.Generate(context.Background(), "rubric", messages, nil)
	if err != nil {
		t.Fatal(err)
	}
	messages = copilotNext(messages, response, textBlock(noToolCallNudge))
	if _, err := m.Generate(context.Background(), "rubric", messages, nil); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(s.prompts, []string{"diff", noToolCallNudge}) {
		t.Fatalf("replayed history or lost nudge: %v", s.prompts)
	}
}

func TestCopilotRejectsChangedHistoryAndInvalidResults(t *testing.T) {
	for _, kind := range []string{"history", "options", "missing", "duplicate", "unknown", "text", "results before call"} {
		t.Run(kind, func(t *testing.T) {
			s := newFakeCopilotSession()
			m := fakeCopilotModel(s)
			s.onSend = func(string) { s.call("preview", "preview_plan", json.RawMessage(`{}`)) }
			tools := (&toolbox{}).tools()
			messages := copilotUser("diff")
			if kind == "results before call" {
				messages[0].Content = []Block{{Type: "tool_result", ToolUseID: "preview"}}
				if _, err := m.Generate(context.Background(), "rubric", messages, tools); err == nil {
					t.Fatal("accepted initial tool result")
				}
				return
			}
			response, err := m.Generate(context.Background(), "rubric", messages, tools)
			if err != nil {
				t.Fatal(err)
			}
			result := Block{Type: "tool_result", ToolUseID: "preview", ToolResult: "valid"}
			messages = copilotNext(messages, response, result)
			system := "rubric"
			switch kind {
			case "history":
				messages[0].Content = []Block{textBlock("different diff")}
			case "options":
				system = "different rubric"
			case "missing":
				messages[len(messages)-1].Content = nil
			case "duplicate":
				messages[len(messages)-1].Content = []Block{result, result}
			case "unknown":
				messages[len(messages)-1].Content[0].ToolUseID = "unknown"
			case "text":
				messages[len(messages)-1].Content = []Block{textBlock("fake result")}
			}
			if _, err := m.Generate(context.Background(), system, messages, tools); err == nil {
				t.Fatal("accepted invalid continuation")
			}
			if len(s.results) != 0 || s.closed != 1 {
				t.Fatalf("must validate whole batch then stop: results=%v closed=%d", s.results, s.closed)
			}
			m.Close()
			if s.closed != 1 {
				t.Fatal("Close must be idempotent")
			}
		})
	}
}

func TestCopilotResponseFailures(t *testing.T) {
	length := "length"
	for _, tc := range []struct {
		name   string
		events []copilot.SessionEventData
		want   string
	}{
		{"quota", []copilot.SessionEventData{&copilot.SessionErrorData{ErrorType: "quota", Message: "quota exceeded"}}, "quota exceeded"},
		{"idle", []copilot.SessionEventData{&copilot.SessionIdleData{}}, "without a complete response"},
		{"truncated", []copilot.SessionEventData{&copilot.AssistantUsageData{FinishReason: &length}}, "length"},
		{"permission", []copilot.SessionEventData{&copilot.PermissionRequestedData{}}, "permissions outside"},
		{"abort", []copilot.SessionEventData{&copilot.AbortData{}}, "aborted"},
		{"unknown tool", []copilot.SessionEventData{&copilot.AssistantMessageData{ToolRequests: []rpc.AssistantMessageToolRequest{{Name: "shell", ToolCallID: "x"}}}}, "unexpected"},
		{"unknown dispatch", []copilot.SessionEventData{&copilot.ExternalToolRequestedData{ToolName: "shell", ToolCallID: "x", RequestID: "r"}}, "unexpected"},
		{"mismatched dispatch", []copilot.SessionEventData{
			&copilot.AssistantMessageData{ToolRequests: []rpc.AssistantMessageToolRequest{{Name: "submit", ToolCallID: "x"}}},
			&copilot.ExternalToolRequestedData{ToolName: "preview_plan", ToolCallID: "x", RequestID: "r"},
		}, "does not match"},
		{"incomplete dispatch", []copilot.SessionEventData{
			&copilot.AssistantMessageData{ToolRequests: []rpc.AssistantMessageToolRequest{{Name: "submit", ToolCallID: "x"}}},
			&copilot.SessionIdleData{},
		}, "without a complete response"},
		{"closed", nil, "session ended"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newFakeCopilotSession()
			for _, event := range tc.events {
				s.emit(event)
			}
			close(s.events)
			m := fakeCopilotModel(s)
			_, err := m.Generate(context.Background(), "rubric", copilotUser("diff"), (&toolbox{}).tools())
			if err == nil || !strings.Contains(err.Error(), tc.want) || s.closed != 1 {
				t.Fatalf("error=%v closed=%d, want %q and stopped session", err, s.closed, tc.want)
			}
		})
	}
}

func TestCopilotCancellationAndTransportErrors(t *testing.T) {
	t.Run("cancel", func(t *testing.T) {
		s := newFakeCopilotSession()
		m := fakeCopilotModel(s)
		ctx, cancel := context.WithCancel(context.Background())
		s.onSend = func(string) { cancel() }
		_, err := m.Generate(ctx, "rubric", copilotUser("diff"), nil)
		if !errors.Is(err, context.Canceled) || s.closed != 1 {
			t.Fatalf("error=%v closed=%d", err, s.closed)
		}
	})
	t.Run("send", func(t *testing.T) {
		s := newFakeCopilotSession()
		s.sendErr = errors.New("broken connection")
		_, err := fakeCopilotModel(s).Generate(context.Background(), "rubric", copilotUser("diff"), nil)
		if err == nil || !strings.Contains(err.Error(), "broken connection") || s.closed != 1 {
			t.Fatalf("error=%v closed=%d", err, s.closed)
		}
	})
	t.Run("missing CLI", func(t *testing.T) {
		t.Setenv("COPILOT_PROVIDER_BASE_URL", "")
		m := &CopilotModel{Model: "gpt-6.1-sol", CLIPath: filepath.Join(t.TempDir(), "nonexistent")}
		_, err := m.Generate(context.Background(), "rubric", copilotUser("diff"), nil)
		if err == nil || !strings.Contains(err.Error(), "install it and run `copilot login`") {
			t.Fatalf("error=%v", err)
		}
	})
	t.Run("BYOK", func(t *testing.T) {
		t.Setenv("COPILOT_PROVIDER_BASE_URL", "https://example.invalid")
		_, _, _, err := startCopilotClient(context.Background(), "copilot")
		if err == nil || !strings.Contains(err.Error(), "instead of BYOK") {
			t.Fatalf("error=%v", err)
		}
	})
}

func TestCopilotAbridgeOwnsValidationAndLifecycle(t *testing.T) {
	const diff = "diff --git a/a.go b/a.go\n--- a/a.go\n+++ b/a.go\n@@ -1 +1 @@\n-old()\n+new()\n"
	for _, outcome := range []string{"success", "turn limit", "error", "fallback"} {
		t.Run(outcome, func(t *testing.T) {
			var sessions []*fakeCopilotSession
			model := &CopilotModel{Model: "gpt-6.1-sol", open: func(_ context.Context, options copilotOptions) (copilotSession, error) {
				s := newFakeCopilotSession()
				sessions = append(sessions, s)
				if options.System != systemPrompt || len(options.Tools) != 2 {
					t.Fatal("must use meat's rubric/tools")
				}
				s.onSend = func(string) {
					if outcome == "error" {
						s.emit(&copilot.SessionErrorData{ErrorType: "quota", Message: "quota exceeded"})
						return
					}
					if outcome == "turn limit" {
						s.emit(&copilot.AssistantMessageData{Content: "Thinking."})
						s.emit(&copilot.SessionIdleData{})
						return
					}
					// Keeping all rows triggers meat's refinement/fallback path.
					if outcome == "fallback" {
						s.call("submit", "submit", json.RawMessage(`{"remove":[],"replace":[],"fold":[],"summary":"Changes the call."}`))
						return
					}
					// An invalid coordinate must come back as a tool error, not
					// bypass meat's compiler or become a reading diff.
					s.call("bad", "submit", json.RawMessage(`{"remove":[{"start_line":999,"end_line":999}],"replace":[],"fold":[],"summary":"Bad plan."}`))
				}
				s.onToolResult = func(id string, block Block) {
					if outcome == "fallback" {
						s.emit(&copilot.SessionErrorData{ErrorType: "quota", Message: "quota exceeded"})
						return
					}
					if id != "request-bad" || !block.ToolError {
						t.Fatalf("invalid submission escaped validation: id=%s result=%+v", id, block)
					}
					s.call("good", "submit", json.RawMessage(`{"remove":[{"start_line":1,"end_line":6}],"replace":[],"fold":[],"summary":"No meaningful change."}`))
				}
				return s, nil
			}}
			// Reuse the original configuration across independent calls.
			for i := 0; i < 2; i++ {
				result, err := Abridge(context.Background(), model, Request{UnifiedDiff: diff, MaxTurns: 2})
				if outcome == "success" && (err != nil || result.SmartDiff != "" || result.Summary != "No meaningful change.") {
					t.Fatalf("result=%+v error=%v", result, err)
				}
				if outcome == "fallback" && (err != nil || result.Summary != "Changes the call.") {
					t.Fatalf("fallback=%+v error=%v", result, err)
				}
				if (outcome == "turn limit" || outcome == "error") && err == nil {
					t.Fatal("expected failure")
				}
				if len(sessions) != i+1 || sessions[i].closed != 1 || model.run != nil {
					t.Fatalf("conversation not scoped/closed: sessions=%d closed=%d", len(sessions), sessions[i].closed)
				}
			}
		})
	}
}

func TestCopilotChunkConversationsAreIndependent(t *testing.T) {
	one := fileSection("a", 5)
	defer setSingleRunBudget(t, len(numberedDiff(one)))()
	var sessions []*fakeCopilotSession
	model := &CopilotModel{Model: "gpt-6.1-sol", open: func(context.Context, copilotOptions) (copilotSession, error) {
		if len(sessions) != 0 && sessions[len(sessions)-1].closed != 1 {
			t.Fatal("previous chunk session must be closed before starting the next")
		}
		s := newFakeCopilotSession()
		s.onSend = func(string) {
			// Every chunk deliberately reuses the same tool-call ID. State
			// from another chunk must never supply or consume this result.
			s.call("submit", "submit", json.RawMessage(`{"remove":[{"start_line":1,"end_line":9}],"replace":[],"fold":[],"summary":"No meaningful change."}`))
		}
		sessions = append(sessions, s)
		return s, nil
	}}
	result, err := Abridge(context.Background(), model, Request{UnifiedDiff: one + fileSection("b", 5) + fileSection("c", 5), MaxTurns: 1})
	if err != nil || result.SmartDiff != "" || len(sessions) != 3 {
		t.Fatalf("result=%+v sessions=%d error=%v", result, len(sessions), err)
	}
	for _, session := range sessions {
		if session.closed != 1 || len(session.prompts) != 1 || len(session.results) != 0 {
			t.Fatalf("chunk state leaked: %+v", session)
		}
	}
}

// This is deliberately opt-in: it uses the saved Copilot login and real credits.
func TestCopilotE2E(t *testing.T) {
	if os.Getenv("MEAT_COPILOT_E2E") != "1" {
		t.Skip("set MEAT_COPILOT_E2E=1 to use the Copilot subscription")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	model, err := NewCopilotFromEnv("gpt-6.1-sol")
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	const system = "You are testing a model adapter. First call read_file with path alpha. After receiving its result, call submit with the exact returned text in summary. Do not call any other tools."
	tools := []Tool{
		{Name: "read_file", Description: "Read a test value", InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`)},
		{Name: "submit", Description: "Submit the test value", InputSchema: json.RawMessage(`{"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"]}`)},
	}
	messages := copilotUser("Run the adapter test now.")
	response, err := model.Generate(ctx, system, messages, tools)
	if err != nil {
		t.Fatal(err)
	}
	var results []Block
	for _, block := range response.Content {
		if block.Type == "tool_use" {
			if block.ToolName != "read_file" {
				t.Fatalf("expected read_file, got %s", block.ToolName)
			}
			results = append(results, Block{Type: "tool_result", ToolUseID: block.ID, ToolResult: "MEAT_COPILOT_ROUNDTRIP_OK"})
		}
	}
	if len(results) != 1 {
		t.Fatalf("expected one read_file call: %+v", response)
	}
	messages = copilotNext(messages, response, results...)
	response, err = model.Generate(ctx, system, messages, tools)
	if err != nil {
		t.Fatal(err)
	}
	var found bool
	for _, block := range response.Content {
		if block.Type == "tool_use" && block.ToolName == "submit" {
			var input struct {
				Summary string `json:"summary"`
			}
			if err := json.Unmarshal(block.ToolInput, &input); err != nil || input.Summary != "MEAT_COPILOT_ROUNDTRIP_OK" {
				t.Fatalf("tool result was not preserved: %s", block.ToolInput)
			}
			found = true
		}
	}
	if !found {
		t.Fatalf("expected submit: %+v", response)
	}
	t.Logf("%s: live read_file -> tool result -> submit round trip succeeded", model.Model)
}

func ExampleResolveModel_copilot() {
	fmt.Println(ResolveModel("copilot/gpt-6.1-sol"))
	// Output: copilot/gpt-6.1-sol
}
