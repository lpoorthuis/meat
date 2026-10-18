package meat

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// DefaultModel is meat's built-in default model.
const DefaultModel = DefaultOpenAIModel

// ResolveModel applies the CLI model fallback chain: the explicit value, then
// $MEAT_MODEL, then DefaultModel. It performs no network or credential work, so
// callers can resolve the cache identity cheaply and offline.
func ResolveModel(model string) string {
	return resolveModel(model, DefaultModel)
}

func resolveModel(model, fallback string) string {
	if model == "" {
		model = os.Getenv("MEAT_MODEL")
	}
	if model == "" {
		model = fallback
	}
	return model
}

// NewModelFromEnv constructs the built-in backend appropriate for model.
// google/* and legacy vertex/* model IDs use Google Cloud credentials, Claude
// model IDs use direct Anthropic Messages, and all other IDs use OpenAI Responses.
func NewModelFromEnv(ctx context.Context, model string) (Model, error) {
	model = ResolveModel(model)
	if isVertexModel(model) {
		_, vertexModel, _ := strings.Cut(model, "/")
		if vertexModel == "" {
			return nil, fmt.Errorf("Google Cloud model id is empty")
		}
		if isAnthropicModel(vertexModel) {
			return NewVertexAnthropicFromEnv(ctx, vertexModel)
		}
		return NewGeminiFromEnv(ctx, vertexModel)
	}
	if isAnthropicModel(model) {
		return NewAnthropicFromEnv(ctx, model)
	}
	return NewOpenAIFromEnv(ctx, model)
}

func isAnthropicModel(model string) bool {
	model = strings.TrimPrefix(model, "anthropic/")
	return strings.HasPrefix(model, "claude-")
}
