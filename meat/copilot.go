package meat

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"slices"
	"strings"
	"sync"
	"time"

	copilot "github.com/github/copilot-sdk/go"
	"github.com/github/copilot-sdk/go/rpc"
)

// CopilotModel uses the official Copilot SDK and the CLI's existing login.
// Abridge creates and closes a separate conversation for each run/chunk. Direct
// Generate callers must keep one linear conversation and call Close when done.
// Copilot never executes meat's tools: calls are handed back to meat's loop.
type CopilotModel struct {
	Model           string // required, unprefixed Copilot model ID
	CLIPath         string
	ReasoningEffort string // defaults to medium for models supporting it

	mu   sync.Mutex
	run  *copilotRun
	open func(context.Context, copilotOptions) (copilotSession, error)
}

type copilotOptions struct {
	Model, CLIPath, ReasoningEffort, System string
	Tools                                   []Tool
}

type copilotSession interface {
	Send(context.Context, string) error
	ToolResult(context.Context, string, Block) error
	Events() <-chan copilot.SessionEvent
	Close()
}

type copilotRun struct {
	session     copilotSession
	optionsHash [32]byte
	historyHash [32]byte
	historyLen  int
	pending     map[string]string // model tool-call ID -> SDK pending request ID
}

// NewCopilotFromEnv configures a lazy SDK provider for an explicit, unprefixed
// model ID. MEAT_COPILOT_CLI overrides COPILOT_CLI_PATH; otherwise copilot is
// resolved from PATH. Authentication stays
// with the CLI, including COPILOT_GITHUB_TOKEN/GH_TOKEN/GITHUB_TOKEN precedence.
func NewCopilotFromEnv(model string) (*CopilotModel, error) {
	if err := validateCopilotModelID(model); err != nil {
		return nil, err
	}
	return &CopilotModel{Model: model, CLIPath: copilotCLIPath()}, nil
}

func validateCopilotModelID(model string) error {
	if model == "" {
		return fmt.Errorf("Copilot model id is empty; select copilot/<model>")
	}
	if strings.TrimSpace(model) != model || strings.ContainsAny(model, "/\r\n\t") {
		return fmt.Errorf("invalid Copilot model id %q", model)
	}
	return nil
}

func copilotCLIPath() string {
	if path := os.Getenv("MEAT_COPILOT_CLI"); path != "" {
		return path
	}
	return os.Getenv("COPILOT_CLI_PATH")
}

// newConversation scopes SDK state and cleanup to a single Abridge run. The
// original model remains a reusable configuration, including for concurrent runs.
func (m *CopilotModel) newConversation() (Model, func()) {
	conversation := &CopilotModel{Model: m.Model, CLIPath: m.CLIPath, ReasoningEffort: m.ReasoningEffort, open: m.open}
	return conversation, conversation.Close
}

// Close stops the runtime, deletes its session, and removes temporary files.
func (m *CopilotModel) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closeRun()
}

func (m *CopilotModel) closeRun() {
	if m.run != nil {
		m.run.session.Close()
		m.run = nil
	}
}

// Generate translates one SDK model response into meat's text/tool_use blocks.
// On the next turn it supplies meat's tool results to the pending SDK calls,
// preserving the live session (and its encrypted reasoning) without replaying or
// flattening the conversation into a prompt.
func (m *CopilotModel) Generate(ctx context.Context, system string, messages []Message, tools []Tool) (_ *Response, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	defer func() {
		if err != nil {
			m.closeRun() // errors/cancellation must not leave a running agent behind
		}
	}()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateCopilotModelID(m.Model); err != nil {
		return nil, err
	}
	options := copilotOptions{Model: m.Model, CLIPath: m.CLIPath, ReasoningEffort: m.ReasoningEffort, System: system, Tools: tools}
	optionsHash, err := copilotHash(options)
	if err != nil {
		return nil, fmt.Errorf("encode Copilot options: %w", err)
	}
	if len(messages) == 0 || messages[len(messages)-1].Role != RoleUser {
		return nil, fmt.Errorf("Copilot requires a user message")
	}
	if m.run == nil {
		if len(messages) != 1 {
			return nil, fmt.Errorf("Copilot requires a new conversation; cannot resume an arbitrary transcript")
		}
		prompt, err := copilotPrompt(messages[0])
		if err != nil {
			return nil, err
		}
		open := m.open
		if open == nil {
			open = openCopilotSession
		}
		session, err := open(ctx, options)
		if err != nil {
			return nil, err
		}
		m.run = &copilotRun{session: session, optionsHash: optionsHash}
		if err := session.Send(ctx, prompt); err != nil {
			return nil, fmt.Errorf("send Copilot prompt: %w", err)
		}
	} else {
		run := m.run
		if len(messages) != run.historyLen+1 || optionsHash != run.optionsHash {
			return nil, fmt.Errorf("Copilot conversation changed; start a new conversation")
		}
		historyHash, err := copilotHash(messages[:run.historyLen])
		if err != nil || historyHash != run.historyHash {
			return nil, fmt.Errorf("Copilot conversation history changed; start a new conversation")
		}
		last := messages[len(messages)-1]
		if len(run.pending) == 0 {
			prompt, err := copilotPrompt(last)
			if err != nil {
				return nil, err
			}
			if err := run.session.Send(ctx, prompt); err != nil {
				return nil, fmt.Errorf("send Copilot prompt: %w", err)
			}
		} else {
			// Validate the entire batch before unblocking any pending tool.
			seen := make(map[string]bool)
			for _, block := range last.Content {
				if block.Type != "tool_result" || run.pending[block.ToolUseID] == "" || seen[block.ToolUseID] {
					return nil, fmt.Errorf("Copilot received an unexpected or duplicate tool result %q", block.ToolUseID)
				}
				seen[block.ToolUseID] = true
			}
			if len(seen) != len(run.pending) {
				return nil, fmt.Errorf("Copilot requires results for all %d pending tools", len(run.pending))
			}
			for _, block := range last.Content {
				if err := run.session.ToolResult(ctx, run.pending[block.ToolUseID], block); err != nil {
					return nil, fmt.Errorf("return Copilot tool result: %w", err)
				}
			}
			run.pending = nil
		}
	}

	response, pending, err := receiveCopilotResponse(ctx, m.run.session.Events(), tools)
	if err != nil {
		return nil, err
	}
	history := append(slices.Clone(messages), Message{Role: RoleAssistant, Content: response.Content})
	m.run.historyHash, err = copilotHash(history)
	if err != nil {
		return nil, fmt.Errorf("encode Copilot conversation: %w", err)
	}
	m.run.historyLen = len(history)
	m.run.pending = pending
	return response, nil
}

func copilotHash(value any) ([32]byte, error) {
	encoded, err := json.Marshal(value)
	return sha256.Sum256(encoded), err
}

func copilotPrompt(message Message) (string, error) {
	var parts []string
	for _, block := range message.Content {
		if block.Type != "text" {
			return "", fmt.Errorf("Copilot expected text, got %q", block.Type)
		}
		parts = append(parts, block.Text)
	}
	if len(parts) == 0 {
		return "", fmt.Errorf("Copilot received an empty user message")
	}
	return strings.Join(parts, "\n\n"), nil
}

func receiveCopilotResponse(ctx context.Context, events <-chan copilot.SessionEvent, tools []Tool) (*Response, map[string]string, error) {
	allowed := make(map[string]bool, len(tools))
	for _, tool := range tools {
		allowed[tool.Name] = true
	}
	response := &Response{}
	calls := make(map[string]string) // model tool-call ID -> tool name
	requests := make(map[string]*copilot.ExternalToolRequestedData)
	var messageSeen, messageComplete bool
	for {
		select {
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		case event, ok := <-events:
			if !ok {
				return nil, nil, fmt.Errorf("Copilot session ended before a response")
			}
			if event.AgentID != nil && *event.AgentID != "" {
				continue
			}
			switch data := event.Data.(type) {
			case *copilot.AssistantUsageData:
				if data.InputTokens != nil {
					response.InputTokens += int(*data.InputTokens)
				}
				if data.OutputTokens != nil {
					response.OutputTokens += int(*data.OutputTokens)
				}
				if data.FinishReason != nil && (*data.FinishReason == "length" || *data.FinishReason == "content_filter") {
					return nil, nil, fmt.Errorf("Copilot response stopped with %s", *data.FinishReason)
				}
			case *copilot.AssistantMessageData:
				messageSeen = true
				messageComplete = data.ChunkCount == nil
				if data.ChunkCount != nil {
					if data.ChunkIndex == nil || *data.ChunkCount <= 0 || *data.ChunkIndex < 0 || *data.ChunkIndex >= *data.ChunkCount {
						return nil, nil, fmt.Errorf("Copilot returned invalid response chunk coordinates")
					}
					messageComplete = *data.ChunkIndex == *data.ChunkCount-1
				}
				if data.Content != "" {
					response.Content = append(response.Content, textBlock(data.Content))
				}
				for _, call := range data.ToolRequests {
					if !allowed[call.Name] || call.ToolCallID == "" || calls[call.ToolCallID] != "" {
						return nil, nil, fmt.Errorf("Copilot requested an unexpected or duplicate tool %q", call.Name)
					}
					arguments, err := json.Marshal(call.Arguments)
					if err != nil {
						return nil, nil, fmt.Errorf("encode Copilot tool arguments: %w", err)
					}
					calls[call.ToolCallID] = call.Name
					response.Content = append(response.Content, Block{Type: "tool_use", ID: call.ToolCallID, ToolName: call.Name, ToolInput: arguments})
				}
			case *copilot.ExternalToolRequestedData:
				if !allowed[data.ToolName] || data.RequestID == "" || data.ToolCallID == "" || requests[data.ToolCallID] != nil {
					return nil, nil, fmt.Errorf("Copilot dispatched an unexpected or duplicate tool %q", data.ToolName)
				}
				requests[data.ToolCallID] = data
			case *copilot.SessionIdleData:
				if !messageSeen || !messageComplete || len(calls) != 0 || len(requests) != 0 {
					return nil, nil, fmt.Errorf("Copilot became idle without a complete response")
				}
				return response, nil, nil
			case *copilot.SessionErrorData:
				return nil, nil, fmt.Errorf("Copilot %s: %s", data.ErrorType, data.Message)
			case *copilot.PermissionRequestedData:
				return nil, nil, fmt.Errorf("Copilot requested permissions outside meat's tools")
			case *copilot.AbortData:
				return nil, nil, fmt.Errorf("Copilot response was aborted")
			}
			if messageComplete && len(calls) != 0 && len(requests) == len(calls) {
				pending := make(map[string]string, len(calls))
				for id, name := range calls {
					request := requests[id]
					if request == nil || request.ToolName != name {
						return nil, nil, fmt.Errorf("Copilot tool dispatch does not match response")
					}
					pending[id] = request.RequestID
				}
				return response, pending, nil
			}
		}
	}
}

// ListCopilotModels discovers enabled model IDs using the CLI's current login.
func ListCopilotModels(ctx context.Context) ([]string, error) {
	client, _, cleanup, err := startCopilotClient(ctx, copilotCLIPath())
	if err != nil {
		return nil, err
	}
	defer cleanup()
	models, err := client.ListModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("list Copilot models: %w", err)
	}
	var ids []string
	for _, model := range models {
		if model.Policy == nil || model.Policy.State != "disabled" {
			ids = append(ids, model.ID)
		}
	}
	slices.Sort(ids)
	return ids, nil
}

func startCopilotClient(ctx context.Context, cliPath string) (*copilot.Client, string, func(), error) {
	if os.Getenv("COPILOT_PROVIDER_BASE_URL") != "" {
		return nil, "", nil, fmt.Errorf("unset COPILOT_PROVIDER_BASE_URL to use the Copilot subscription instead of BYOK")
	}
	if cliPath == "" {
		cliPath = "copilot"
	}
	path, err := exec.LookPath(cliPath)
	if err != nil {
		return nil, "", nil, fmt.Errorf("find Copilot CLI (install it and run `copilot login`, or set MEAT_COPILOT_CLI): %w", err)
	}
	dir, err := os.MkdirTemp("", "meat-copilot-")
	if err != nil {
		return nil, "", nil, err
	}
	client := copilot.NewClient(copilotClientOptions(path, dir))
	cleanup := func() {
		client.ForceStop()
		os.RemoveAll(dir)
	}
	if err := client.Start(ctx); err != nil {
		cleanup()
		return nil, "", nil, fmt.Errorf("start Copilot SDK: %w", err)
	}
	auth, err := client.GetAuthStatus(ctx)
	if err != nil || !auth.IsAuthenticated {
		cleanup()
		if err != nil {
			return nil, "", nil, fmt.Errorf("check Copilot login: %w", err)
		}
		return nil, "", nil, fmt.Errorf("Copilot is not authenticated; run `copilot login`")
	}
	return client, dir, cleanup, nil
}

func copilotClientOptions(path, dir string) *copilot.ClientOptions {
	return &copilot.ClientOptions{
		Connection: copilot.StdioConnection{Path: path, Args: []string{"--no-auto-update", "--disable-builtin-mcps", "--no-custom-instructions"}},
		// Do not set BaseDirectory or use ModeEmpty: both can change credential
		// discovery and silently fall back from the Copilot login to `gh`.
		Mode:             copilot.ModeCopilotCli,
		WorkingDirectory: dir,
		LogLevel:         "error",
	}
}

func copilotSessionConfig(options copilotOptions, dir string) (*copilot.SessionConfig, error) {
	config := &copilot.SessionConfig{
		ClientName:                         "meat",
		Model:                              options.Model,
		SystemMessage:                      &copilot.SystemMessageConfig{Mode: "replace", Content: options.System},
		ConfigDirectory:                    dir, // isolate configuration, not client credentials
		WorkingDirectory:                   dir,
		AvailableTools:                     []string{}, // explicit deny-all when meat has no tools
		EnableConfigDiscovery:              copilot.Bool(false),
		SkipCustomInstructions:             copilot.Bool(true),
		EnableOnDemandInstructionDiscovery: copilot.Bool(false),
		EnableFileHooks:                    copilot.Bool(false),
		EnableHostGitOperations:            copilot.Bool(false),
		EnableSessionStore:                 copilot.Bool(false),
		EnableGitHubCacheStore:             copilot.Bool(false),
		EnableSkills:                       copilot.Bool(false),
		EnableSessionTelemetry:             copilot.Bool(false),
		Memory:                             &copilot.MemoryConfiguration{Enabled: false},
		InfiniteSessions:                   &copilot.InfiniteSessionConfig{Enabled: copilot.Bool(false)},
	}
	seen := make(map[string]bool)
	for _, tool := range options.Tools {
		if tool.Name == "" || seen[tool.Name] {
			return nil, fmt.Errorf("empty or duplicate Copilot tool name %q", tool.Name)
		}
		seen[tool.Name] = true
		var parameters map[string]any
		if err := json.Unmarshal(tool.InputSchema, &parameters); err != nil || parameters == nil {
			return nil, fmt.Errorf("invalid JSON schema for Copilot tool %q", tool.Name)
		}
		config.Tools = append(config.Tools, copilot.Tool{
			Name: tool.Name, Description: tool.Description, Parameters: parameters,
			OverridesBuiltInTool: true, SkipPermission: true, Defer: copilot.ToolDeferNever,
			// No handler: meat executes the call and returns its validated result.
		})
		config.AvailableTools = append(config.AvailableTools, "custom:"+tool.Name)
	}
	return config, nil
}

func openCopilotSession(ctx context.Context, options copilotOptions) (copilotSession, error) {
	// Validate tool declarations before starting a child process.
	config, err := copilotSessionConfig(options, "")
	if err != nil {
		return nil, err
	}
	client, dir, cleanup, err := startCopilotClient(ctx, options.CLIPath)
	if err != nil {
		return nil, err
	}
	models, err := client.ListModels(ctx)
	if err != nil {
		cleanup()
		return nil, fmt.Errorf("list Copilot models: %w", err)
	}
	var selected *copilot.ModelInfo
	for i := range models {
		if models[i].ID == options.Model && (models[i].Policy == nil || models[i].Policy.State != "disabled") {
			selected = &models[i]
			break
		}
	}
	if selected == nil {
		cleanup()
		return nil, fmt.Errorf("Copilot model %q is unavailable for the current login; run `meat -list-models copilot`", options.Model)
	}
	if selected.Capabilities.Supports.ReasoningEffort {
		config.ReasoningEffort = options.ReasoningEffort
		if config.ReasoningEffort == "" {
			config.ReasoningEffort = DefaultReasoningEffort
		}
		if len(selected.SupportedReasoningEfforts) > 0 && !slices.Contains(selected.SupportedReasoningEfforts, config.ReasoningEffort) {
			cleanup()
			return nil, fmt.Errorf("Copilot model %q does not support reasoning effort %q", options.Model, config.ReasoningEffort)
		}
	} else if options.ReasoningEffort != "" {
		cleanup()
		return nil, fmt.Errorf("Copilot model %q does not support reasoning effort", options.Model)
	}
	config.ConfigDirectory, config.WorkingDirectory = dir, dir
	session, err := client.CreateSession(ctx, config)
	if err != nil {
		cleanup()
		return nil, fmt.Errorf("create Copilot session: %w", err)
	}
	transport := &sdkCopilotSession{client: client, session: session, events: make(chan copilot.SessionEvent, 64), done: make(chan struct{}), cleanup: cleanup}
	transport.unsubscribe = session.On(func(event copilot.SessionEvent) {
		switch event.Data.(type) {
		case *copilot.AssistantMessageData, *copilot.AssistantUsageData, *copilot.ExternalToolRequestedData,
			*copilot.SessionIdleData, *copilot.SessionErrorData, *copilot.PermissionRequestedData, *copilot.AbortData:
			select {
			case transport.events <- event:
			case <-transport.done:
			}
		}
	})
	return transport, nil
}

type sdkCopilotSession struct {
	client      *copilot.Client
	session     *copilot.Session
	events      chan copilot.SessionEvent
	done        chan struct{}
	unsubscribe func()
	cleanup     func()
}

func (s *sdkCopilotSession) Events() <-chan copilot.SessionEvent { return s.events }

func (s *sdkCopilotSession) Send(ctx context.Context, prompt string) error {
	_, err := s.session.Send(ctx, copilot.MessageOptions{Prompt: prompt})
	return err
}

func (s *sdkCopilotSession) ToolResult(ctx context.Context, requestID string, block Block) error {
	resultType := "success"
	if block.ToolError {
		resultType = "failure"
	}
	_, err := s.session.RPC.Tools.HandlePendingToolCall(ctx, &rpc.HandlePendingToolCallRequest{
		RequestID: requestID,
		Result: rpc.ExternalToolTextResultForLlm{
			TextResultForLlm: block.ToolResult, ResultType: &resultType,
		},
	})
	return err
}

func (s *sdkCopilotSession) Close() {
	close(s.done)
	s.unsubscribe()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = s.session.Abort(ctx)
	_ = s.client.DeleteSession(ctx, s.session.SessionID)
	s.cleanup()
}
