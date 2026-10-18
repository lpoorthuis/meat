package meat

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"os"
	"path/filepath"
	"testing"
)

func TestGoogleProjectFromServiceAccount(t *testing.T) {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(map[string]string{
		"type":         "service_account",
		"project_id":   "key-project",
		"client_email": "test@key-project.iam.gserviceaccount.com",
		"private_key":  string(pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})),
		"token_uri":    "https://oauth2.googleapis.com/token",
	})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "key.json")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(meatGoogleCredentialsEnv, path)
	t.Setenv("GOOGLE_CLOUD_PROJECT", "unrelated-project")
	for _, explicit := range []string{"", "override-project"} {
		t.Run("explicit="+explicit, func(t *testing.T) {
			t.Setenv("MEAT_GOOGLE_CLOUD_PROJECT", explicit)
			want := explicit
			if want == "" {
				want = "key-project"
			}
			project, err := googleProject(context.Background())
			if err != nil || project != want {
				t.Fatalf("project = %q, error = %v; want %q", project, err, want)
			}
			for _, model := range []string{"google/gemini-4-argon", "google/claude-sonnet-4@20250514"} {
				if _, err := NewModelFromEnv(context.Background(), model); err != nil {
					t.Errorf("%s: %v", model, err)
				}
			}
		})
	}
}
