package meat

import (
	"context"
	"strings"
	"testing"
)

func TestVertexRequiresMeatProject(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "unrelated-project")
	t.Setenv("MEAT_GOOGLE_CLOUD_PROJECT", "")
	t.Setenv(meatGoogleCredentialsEnv, "")
	for _, model := range []string{
		"google/gemini-4-argon",
		"google/claude-sonnet-4@20250514",
		"vertex/gemini-2.5-pro",
		"vertex/claude-sonnet-4@20250514",
	} {
		t.Run(model, func(t *testing.T) {
			_, err := NewModelFromEnv(context.Background(), model)
			if err == nil || !strings.Contains(err.Error(), "set MEAT_GOOGLE_CLOUD_PROJECT") {
				t.Fatalf("error = %v, want missing meat-specific project", err)
			}
		})
	}
}
