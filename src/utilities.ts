import { basename } from "path";
import * as vscode from "vscode";
import { ALLOWED_COMPLETION_MODELS } from "./constants";
import { ILanguageConfig } from "./interfaces";
import { logger } from "./logger";

/**
 * Set the context value for a given key in the extension context
 * @param {string} key - The context key
 * @param {any} value - The context value
 * @returns {Promise<void>} The promise
 */
export const setContext = async (key: string, value: boolean) => {
  await vscode.commands.executeCommand("setContext", `flexpilot:${key}`, value);
  logger.debug(`Set context: ${key} = ${value}`);
};

/**
 * Get the completion model metadata for a given model
 * @param {string} model - The model name
 * @returns {Promise<(typeof ALLOWED_COMPLETION_MODELS)[number] | undefined>} The completion model metadata
 */
export const getCompletionModelMetadata = (
  model: string,
): (typeof ALLOWED_COMPLETION_MODELS)[number] | undefined => {
  logger.debug("Searching for completion models");
  return ALLOWED_COMPLETION_MODELS.find((metadata) =>
    new RegExp(metadata.regex).test(model),
  );
};

/**
 * Get the GitHub sessionu
 * @returns {Promise<vscode.AuthenticationSession>} The GitHub session
 */
export const getGithubSession =
  async (): Promise<vscode.AuthenticationSession> => {
    logger.debug("Getting GitHub session");
    const session = await vscode.authentication.getSession("github", []);
    if (session) {
      logger.debug("GitHub session found");
      return session;
    }
    logger.notifyError("GitHub session not found");
    throw new Error("GitHub session not found");
  };

/**
 * Get the terminal type from the terminal object
 * @param {vscode.Terminal} terminal - The terminal object
 * @returns {string} The type of terminal
 */
export const getTerminalType = (terminal: vscode.Terminal): string => {
  logger.debug("Getting terminal type");
  if (
    terminal &&
    "shellPath" in terminal.creationOptions &&
    terminal.creationOptions.shellPath
  ) {
    const shellName = basename(terminal.creationOptions.shellPath);
    switch (true) {
      case shellName === "bash.exe":
        logger.debug("Terminal type: Git Bash");
        return "Git Bash";
      case shellName.startsWith("pwsh"):
      case shellName.startsWith("powershell"):
        logger.debug("Terminal type: powershell");
        return "powershell";
      case Boolean(shellName.trim()):
        logger.debug(`Terminal type: ${shellName.split(".")[0]}`);
        return shellName.split(".")[0];
    }
  }
  const defaultType = process.platform === "win32" ? "powershell" : "bash";
  logger.debug(`Returning terminal type: ${defaultType}`);
  return defaultType;
};

/**
 * Get the end of line sequence for a document
 * @param {vscode.TextDocument} document - The document
 * @returns {string} The end of line sequence
 */
export const getEol = (document: vscode.TextDocument): string => {
  logger.debug("Getting end of line sequence for document");
  return document.eol === vscode.EndOfLine.CRLF ? "\r\n" : "\n";
};

/**
 * Retrieves the language configuration for a given language ID.
 * @param {string} languageId - The ID of the language.
 * @returns {ILanguageConfig} The configuration for the specified language.
 */
export const getLanguageConfig = (languageId: string): ILanguageConfig => {
  if (LANGUAGES[languageId]) {
    return LANGUAGES[languageId];
  } else {
    return { markdown: languageId, comment: { start: "//", end: "" } };
  }
};

/**
 * A mapping of language IDs to their respective configurations.
 */
const LANGUAGES: { [key: string]: ILanguageConfig } = {
  abap: {
    markdown: "abap",
    comment: { start: "* ", end: " */" },
  },
  bibtex: {
    markdown: "bibtex",
    comment: { start: "% ", end: "" },
  },
  d: {
    markdown: "d",
    comment: { start: "/* ", end: " */" },
  },
  pascal: {
    markdown: "pascal",
    comment: { start: "{ ", end: " }" },
  },
  erlang: {
    markdown: "erlang",
    comment: { start: "%% ", end: " %%" },
  },
  haml: {
    markdown: "haml",
    comment: { start: "-# ", end: " -#" },
  },
  haskell: {
    markdown: "haskell",
    comment: { start: "{- ", end: " -}" },
  },
  ocaml: {
    markdown: "ocaml",
    comment: { start: "(* ", end: " *)" },
  },
  perl6: {
    markdown: "perl6",
    comment: { start: "/* ", end: " */" },
  },
  sass: {
    markdown: "scss",
    comment: { start: "/* ", end: " */" },
  },
  slim: {
    markdown: "slim",
    comment: { start: "/ ", end: "" },
  },
  stylus: {
    markdown: "stylus",
    comment: { start: "// ", end: "" },
  },
  svelte: {
    markdown: "svelte",
    comment: { start: "/* ", end: " */" },
  },
  vue: {
    markdown: "vue",
    comment: { start: "/* ", end: " */" },
  },
  "vue-html": {
    markdown: "html",
    comment: { start: "<!-- ", end: " -->" },
  },
  razor: {
    markdown: "razor",
    comment: { start: "<!-- ", end: " -->" },
  },
  shaderlab: {
    markdown: "shader",
    comment: { start: "/* ", end: " */" },
  },
  dockerfile: {
    markdown: "dockerfile",
    comment: { start: "# ", end: "" },
  },
  go: {
    markdown: "go",
    comment: { start: "/* ", end: " */" },
  },
  python: {
    markdown: "py",
    comment: { start: '""" ', end: ' """' },
  },
  css: {
    markdown: "css",
    comment: { start: "/* ", end: " */" },
  },
  clojure: {
    markdown: "clj",
    comment: { start: ";; ", end: "" },
  },
  less: {
    markdown: "less",
    comment: { start: "/* ", end: " */" },
  },
  dart: {
    markdown: "dart",
    comment: { start: "/* ", end: " */" },
  },
  tex: {
    markdown: "tex",
    comment: { start: "% ", end: "" },
  },
  latex: {
    markdown: "latex",
    comment: { start: "% ", end: "" },
  },
  scss: {
    markdown: "scss",
    comment: { start: "/* ", end: " */" },
  },
  perl: {
    markdown: "pl",
    comment: { start: "# ", end: "" },
  },
  raku: {
    markdown: "raku",
    comment: { start: "# ", end: "" },
  },
  rust: {
    markdown: "rs",
    comment: { start: "/* ", end: " */" },
  },
  jade: {
    markdown: "pug",
    comment: { start: "//- ", end: "" },
  },
  fsharp: {
    markdown: "fs",
    comment: { start: "(* ", end: " *)" },
  },
  r: {
    markdown: "r",
    comment: { start: "# ", end: "" },
  },
  java: {
    markdown: "java",
    comment: { start: "/* ", end: " */" },
  },
  diff: {
    markdown: "diff",
    comment: { start: "# ", end: " " },
  },
  html: {
    markdown: "html",
    comment: { start: "<!-- ", end: " -->" },
  },
  php: {
    markdown: "php",
    comment: { start: "/* ", end: " */" },
  },
  lua: {
    markdown: "lua",
    comment: { start: "--[[ ", end: " ]]" },
  },
  xml: {
    markdown: "xml",
    comment: { start: "<!-- ", end: " -->" },
  },
  xsl: {
    markdown: "xsl",
    comment: { start: "<!-- ", end: " -->" },
  },
  vb: {
    markdown: "vb",
    comment: { start: "' ", end: "" },
  },
  powershell: {
    markdown: "ps1",
    comment: { start: "<# ", end: " #>" },
  },
  typescript: {
    markdown: "ts",
    comment: { start: "/* ", end: " */" },
  },
  typescriptreact: {
    markdown: "tsx",
    comment: { start: "/* ", end: " */" },
  },
  ini: {
    markdown: "ini",
    comment: { start: "; ", end: " " },
  },
  properties: {
    markdown: "conf",
    comment: { start: "# ", end: " " },
  },
  json: {
    markdown: "json",
    comment: { start: "/* ", end: " */" },
  },
  jsonc: {
    markdown: "jsonc",
    comment: { start: "/* ", end: " */" },
  },
  jsonl: {
    markdown: "jsonl",
    comment: { start: "/* ", end: " */" },
  },
  snippets: {
    markdown: "code-snippets",
    comment: { start: "/* ", end: " */" },
  },
  "git-commit": {
    markdown: "git-commit",
    comment: { start: "# ", end: " " },
  },
  "git-rebase": {
    markdown: "git-rebase",
    comment: { start: "# ", end: " " },
  },
  ignore: {
    markdown: "gitignore_global",
    comment: { start: "# ", end: "" },
  },
  handlebars: {
    markdown: "handlebars",
    comment: { start: "{{!-- ", end: " --}}" },
  },
  c: {
    markdown: "c",
    comment: { start: "/* ", end: " */" },
  },
  cpp: {
    markdown: "cpp",
    comment: { start: "/* ", end: " */" },
  },
  "cuda-cpp": {
    markdown: "cpp",
    comment: { start: "/* ", end: " */" },
  },
  swift: {
    markdown: "swift",
    comment: { start: "/* ", end: " */" },
  },
  makefile: {
    markdown: "mak",
    comment: { start: "# ", end: "" },
  },
  shellscript: {
    markdown: "sh",
    comment: { start: "# ", end: "" },
  },
  markdown: {
    markdown: "md",
    comment: { start: "<!-- ", end: " -->" },
  },
  dockercompose: {
    markdown: "dockercompose",
    comment: { start: "# ", end: "" },
  },
  yaml: {
    markdown: "yaml",
    comment: { start: "# ", end: "" },
  },
  csharp: {
    markdown: "cs",
    comment: { start: "/* ", end: " */" },
  },
  julia: {
    markdown: "jl",
    comment: { start: "#= ", end: " =#" },
  },
  bat: {
    markdown: "bat",
    comment: { start: "@REM ", end: "" },
  },
  groovy: {
    markdown: "groovy",
    comment: { start: "/* ", end: " */" },
  },
  coffeescript: {
    markdown: "coffee",
    comment: { start: "### ", end: " ###" },
  },
  javascriptreact: {
    markdown: "jsx",
    comment: { start: "/* ", end: " */" },
  },
  javascript: {
    markdown: "js",
    comment: { start: "/* ", end: " */" },
  },
  "jsx-tags": {
    markdown: "jsx-tags",
    comment: { start: "{/* ", end: " */}" },
  },
  hlsl: {
    markdown: "hlsl",
    comment: { start: "/* ", end: " */" },
  },
  restructuredtext: {
    markdown: "rst",
    comment: { start: ".. ", end: "" },
  },
  "objective-c": {
    markdown: "m",
    comment: { start: "/* ", end: " */" },
  },
  "objective-cpp": {
    markdown: "cpp",
    comment: { start: "/* ", end: " */" },
  },
  ruby: {
    markdown: "rb",
    comment: { start: "=begin ", end: " =end" },
  },
  sql: {
    markdown: "sql",
    comment: { start: "/* ", end: " */" },
  },
};
