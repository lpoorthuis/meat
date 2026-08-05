# meat

Abridge a code diff into a **reading diff**.

Humans need to review agent-written code in critical systems.
But models are good now. You don't need to review for style or nil-checks
or imports. You need to review concepts, algorithm choices, architecture.

So meat uses a model to reduce a diff to the important parts.
It shows you the meat.

Install with:

```
go install meat.dev/cmd/meat@latest
```

Run with `meat` to review the latest commit.
It takes git-looking parameters to pick commits to review.

By default meat uses OpenAI. It also supports Gemini and Claude through Vertex
AI using [Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc):

```
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="my-project"
export GOOGLE_CLOUD_LOCATION="global" # optional; global is the default

# Gemini through the Google Gen AI SDK
export MEAT_MODEL="vertex/gemini-2.5-pro"
meat

# Claude from Vertex AI Model Garden through the Anthropic SDK
export MEAT_MODEL="vertex/claude-sonnet-4@20250514"
meat
```

The part after `vertex/` is passed to Vertex unchanged, so any enabled text
model with function calling can be selected without a meat release. Models
whose IDs start with `claude-` use Anthropic's Vertex API; other `vertex/*`
models use Vertex GenerateContent. The Vertex AI API and the selected model
must be enabled for the project. Gemini 3.6 Flash is routed to `global`, where
Agent Platform publishes it. Claude Opus 5 and Sonnet 5 are routed to a
multi-region endpoint automatically (`europe-*` → `eu`, `us-*` → `us`; default
`eu`) because they are unavailable on regional endpoints. Other models use
`GOOGLE_CLOUD_LOCATION` unchanged. Claude 429 and transient 5xx responses are
retried with exponential backoff.

It takes a while to process a commit for reading.
So I suggest you have an agent build `meat` into your devtools so that
it pre-processes it.

Very large diffs are split at file and hunk boundaries and abridged
chunk by chunk (up to a few MB), so one huge commit still produces a
single merged reading diff — it just takes proportionally longer.
