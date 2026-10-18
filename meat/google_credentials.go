package meat

import (
	"context"
	"fmt"
	"os"

	"golang.org/x/oauth2/google"
)

const meatGoogleCredentialsEnv = "MEAT_GOOGLE_APPLICATION_CREDENTIALS"

// googleProject prefers an explicit target project, then the project embedded
// in meat's credential file. Do not inherit unrelated Google project settings.
func googleProject(ctx context.Context) (string, error) {
	if project := os.Getenv("MEAT_GOOGLE_CLOUD_PROJECT"); project != "" {
		return project, nil
	}
	if os.Getenv(meatGoogleCredentialsEnv) != "" {
		credentials, err := googleCredentials(ctx)
		if err != nil {
			return "", err
		}
		if credentials.ProjectID != "" {
			return credentials.ProjectID, nil
		}
	}
	return "", fmt.Errorf("no Google Cloud project: set MEAT_GOOGLE_CLOUD_PROJECT or provide project_id in %s", meatGoogleCredentialsEnv)
}

// googleCredentials uses meat's explicit credential file when configured,
// otherwise preserving standard ADC discovery. The SDK refreshes access tokens.
func googleCredentials(ctx context.Context) (*google.Credentials, error) {
	path := os.Getenv(meatGoogleCredentialsEnv)
	if path == "" {
		return google.FindDefaultCredentials(ctx, googleCloudScope)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", meatGoogleCredentialsEnv, err)
	}
	creds, err := google.CredentialsFromJSON(ctx, data, googleCloudScope)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", meatGoogleCredentialsEnv, err)
	}
	return creds, nil
}
