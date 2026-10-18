package meat

import (
	"context"
	"strings"
	"testing"
)

func TestGoogleModelPrefixes(t *testing.T) {
	for _, model := range []string{"google/gemini-4-argon", "google/claude-sonnet-4@20250514", "vertex/gemini-2.5-pro"} {
		if !isVertexModel(model) {
			t.Errorf("%q should route to Google Cloud", model)
		}
	}
	for _, model := range []string{"gpt-4.1", "claude-sonnet-4", "googleish/gemini", "google"} {
		if isVertexModel(model) {
			t.Errorf("%q should not route to Google Cloud", model)
		}
	}
}

func TestEmptyGoogleModelID(t *testing.T) {
	for _, model := range []string{"google/", "vertex/"} {
		_, err := NewModelFromEnv(context.Background(), model)
		if err == nil || !strings.Contains(err.Error(), "model id is empty") {
			t.Errorf("model %q: error = %v, want empty model ID", model, err)
		}
	}
}
