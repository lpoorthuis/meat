package meat

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGoogleCredentialFilePrecedence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "credentials.json")
	// An authorized-user fixture exercises parsing without real secrets or network calls.
	if err := os.WriteFile(path, []byte(`{"type":"authorized_user","client_id":"test-id","client_secret":"test-secret","refresh_token":"test-refresh"}`), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", filepath.Join(dir, "missing-adc.json"))
	t.Setenv(meatGoogleCredentialsEnv, path)
	t.Setenv("MEAT_GOOGLE_CLOUD_PROJECT", "meat-project")
	creds, err := googleCredentials(context.Background())
	if err != nil || creds.TokenSource == nil {
		t.Fatalf("credentials = %v, error = %v", creds, err)
	}
	for _, model := range []string{"google/gemini-4-argon", "google/claude-sonnet-4@20250514"} {
		if _, err := NewModelFromEnv(context.Background(), model); err != nil {
			t.Errorf("%s: %v", model, err)
		}
	}
}

func TestGoogleCredentialFileErrors(t *testing.T) {
	t.Setenv("MEAT_GOOGLE_CLOUD_PROJECT", "meat-project")
	for _, content := range []string{"missing", "invalid-json"} {
		t.Run(content, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "credentials.json")
			if content != "missing" {
				if err := os.WriteFile(path, []byte(content), 0600); err != nil {
					t.Fatal(err)
				}
			}
			t.Setenv(meatGoogleCredentialsEnv, path)
			for _, model := range []string{"google/gemini-4-argon", "google/claude-sonnet-4@20250514"} {
				_, err := NewModelFromEnv(context.Background(), model)
				if err == nil || !strings.Contains(err.Error(), meatGoogleCredentialsEnv) {
					t.Errorf("%s: error = %v, want explicit credential-file error", model, err)
				}
			}
		})
	}
}

func TestGoogleCredentialsFallBackToADC(t *testing.T) {
	t.Setenv(meatGoogleCredentialsEnv, "")
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", filepath.Join(t.TempDir(), "missing-adc.json"))
	if _, err := googleCredentials(context.Background()); err == nil {
		t.Fatal("expected missing standard ADC file error")
	}
}
