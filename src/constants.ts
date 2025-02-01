/**
 * Defines the types of providers available in the application.
 * These types are used to categorize different functionalities.
 */
export const PROVIDER_TYPES = ["chat", "completion"] as const;

/**
 * Defines the models that are allowed to be used for completions.
 * Each model has a regex pattern to match the model name, a context window size,
 * a type, and a tokenizer URL.
 */
export const ALLOWED_COMPLETION_MODELS = [
  {
    regex: "^gpt-3.5-turbo-.*instruct",
    contextWindow: 4000,
    tokenizerUrl:
      "https://cdn.jsdelivr.net/gh/flexpilot-ai/vscode-extension/tokenizers/cl100k_base.json",
  },
  {
    regex: "^codestral-(?!.*mamba)",
    contextWindow: 31500,
    tokenizerUrl:
      "https://cdn.jsdelivr.net/gh/flexpilot-ai/vscode-extension/tokenizers/codestral-v0.1.json",
  },
  {
    regex: "^gpt-35-turbo-.*instruct",
    contextWindow: 4000,
    tokenizerUrl:
      "https://cdn.jsdelivr.net/gh/flexpilot-ai/vscode-extension/tokenizers/cl100k_base.json",
  },
  {
    regex: "^deepseek-.*",
    contextWindow: 128000,
    tokenizerUrl:
      "https://cdn.jsdelivr.net/gh/flexpilot-ai/vscode-extension/tokenizers/deepseek-v3.json",
  },
  {
    regex: "^qwen2\\.5-coder.*-base",
    contextWindow: 32768,
    tokenizerUrl:
      "https://cdn.jsdelivr.net/gh/flexpilot-ai/vscode-extension/tokenizers/qwen2.5-coder.json",
  },
];

/**
 * Defines the various locations or contexts where AI assistance is provided.
 * Each location has a name, description, and associated provider type.
 */
export const LOCATIONS = [
  {
    name: "Chat Suggestions",
    description: "Suggestions that appear in the panel chat",
    type: "chat",
  },
  {
    name: "Chat Title",
    description: "Dynamically generated title for the chat",
    type: "chat",
  },
  {
    name: "Inline Chat",
    description: "Chat used inline inside an active file",
    type: "chat",
  },
  {
    name: "Panel Chat",
    description: "Chat that appears on the panel on the side",
    type: "chat",
  },
  {
    name: "Commit Message",
    description: "Used to generate commit messages for source control",
    type: "chat",
  },
  {
    name: "Inline Completion",
    description: "Completion suggestions in the active file",
    type: "completion",
  },
] as const;

// Note: The 'as const' assertion ensures that the array is read-only
// and its contents are treated as literal types.
