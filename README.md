# meat

Abridge a code diff into a **reading diff**.

Humans need to review agent-written code in critical systems.
But models are good now. You don't need to review for style or nil-checks
or imports. You need to review concepts, algorithm choices, architecture.

So meat uses a model to reduce a diff to the important parts.
It shows you the meat.

Install from a local checkout:

```sh
go install ./cmd/meat
```

Run with `meat` to review the latest commit.
It takes git-looking parameters to pick commits to review.

By default meat uses OpenAI. It also supports Gemini and Claude through Vertex
AI using [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc):

```
gcloud auth application-default login
export MEAT_GOOGLE_CLOUD_PROJECT="my-project"
export MEAT_GOOGLE_CLOUD_LOCATION="global" # optional; global is the default

# Gemini through the Google Gen AI SDK
export MEAT_MODEL="google/gemini-2.5-pro"
meat

# Claude from Vertex AI Model Garden through the Anthropic SDK
export MEAT_MODEL="google/claude-sonnet-4@20250514"
meat
```

Meat uses `MEAT_GOOGLE_CLOUD_PROJECT` and `MEAT_GOOGLE_CLOUD_LOCATION`
(or `MEAT_GOOGLE_CLOUD_REGION` as a location alias), independently of the
unprefixed Google Cloud variables. Authentication uses standard ADC by default.
To use a service account key instead of an interactive login, set:

```sh
export MEAT_GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"
```

This file takes precedence over standard ADC for both Gemini and Claude.
Its `project_id` is used automatically when `MEAT_GOOGLE_CLOUD_PROJECT` is
unset; set that variable only to override the key's project.
Access tokens are refreshed automatically; an invalid or unreadable file fails
without falling back to another identity. Keep the key outside the repository
and restrict its permissions (`chmod 600`). The service account needs access to
the models in `MEAT_GOOGLE_CLOUD_PROJECT`.

Use the `google/` prefix to select Google Cloud; `vertex/` remains supported
as a compatibility alias. The part after either prefix is passed to the API
unchanged, so any enabled text model with function calling can be selected
without a meat release. Models whose IDs start with `claude-` use Anthropic's
Vertex API; other Google models use GenerateContent. The Vertex AI API and the selected model
must be enabled for the project. Gemini 3.6 Flash is routed to `global`, where
Agent Platform publishes it. Claude Opus 5 and Sonnet 5 are routed to a
multi-region endpoint automatically (`europe-*` → `eu`, `us-*` → `us`; default
`eu`) because they are unavailable on regional endpoints. Other models use
`MEAT_GOOGLE_CLOUD_LOCATION` unchanged. Claude 429 and transient 5xx responses are
retried with exponential backoff.

It takes a while to process a commit for reading.
So I suggest you have an agent build `meat` into your devtools so that
it pre-processes it.

Very large diffs are split at file and hunk boundaries and abridged
chunk by chunk (up to a few MB), so one huge commit still produces a
single merged reading diff — it just takes proportionally longer.
