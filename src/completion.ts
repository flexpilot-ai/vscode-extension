import * as vscode from "vscode";
import { logger } from "./logger";
import { ModelProviderManager } from "./providers";
import { statusIcon } from "./status-icon";
import { storage } from "./storage";
import { getLanguageConfig } from "./utilities";

/**
 * InlineCompletionProvider class provides inline completion functionality for the active document.
 * It implements the Singleton pattern to ensure a single instance across the application.
 */
class InlineCompletionProvider implements vscode.InlineCompletionItemProvider {
  private static instance: InlineCompletionProvider | null = null;
  private cacheSuffix: string = "";
  private cachePrefix: string = "";
  private readonly maxOutputTokens = 500;
  private readonly disposable: vscode.Disposable;

  /**
   * Private constructor to prevent direct instantiation.
   * Registers the inline completion item provider.
   */
  private constructor() {
    logger.info("Registering inline completion item provider");
    this.disposable = vscode.languages.registerInlineCompletionItemProvider(
      { pattern: "**/*" },
      this,
    );
    logger.info("Inline completion item provider registered successfully");
  }

  /**
   * Disposes the inline completion provider instance.
   */
  public static dispose(): void {
    if (InlineCompletionProvider.instance) {
      InlineCompletionProvider.instance.disposable.dispose();
      InlineCompletionProvider.instance = null;
    }
    logger.info("Inline completion provider disposed successfully");
  }

  /**
   * Registers the inline completion provider instance.
   */
  public static register() {
    if (!InlineCompletionProvider.instance) {
      InlineCompletionProvider.instance = new InlineCompletionProvider();
      logger.debug("New InlineCompletionProvider instance created");
    }
  }

  /**
   * Provides inline completion items for the current document.
   * @param {vscode.TextDocument} document - The current document.
   * @param {vscode.Position} position - The current position in the document.
   * @param {vscode.InlineCompletionContext} _context - The completion context.
   * @param {vscode.CancellationToken} token - The cancellation token.
   * @returns {Promise<vscode.InlineCompletionList>} A list of inline completions.
   */
  public async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken,
  ): Promise<vscode.InlineCompletionList> {
    if (statusIcon().state === "disabled") {
      logger.debug("Inline completions disabled");
      return new vscode.InlineCompletionList([]);
    }
    const cache = this.checkCompletionCache(document, position);
    if (cache) {
      logger.debug("Returning cached completion");
      return cache;
    }
    return await new Promise<vscode.InlineCompletionList>((resolve) => {
      logger.debug("Debouncing completion request");
      const abortController = new AbortController();
      statusIcon().reset();
      const timeoutId = setTimeout(async () => {
        try {
          logger.debug("Generating completions");
          statusIcon().setLoading();
          const completions = await this.generateCompletions(
            document,
            position,
            abortController,
          );
          resolve(completions);
        } catch (error) {
          if (!abortController.signal.aborted) {
            // Log error if not due to cancellation
            logger.error(error as Error);
            logger.notifyError("Error generating completions");
          }
          resolve(new vscode.InlineCompletionList([]));
        } finally {
          statusIcon().reset();
        }
      }, storage().workspace.get<number>("flexpilot.completions.debounceWait"));

      token.onCancellationRequested(() => {
        clearTimeout(timeoutId);
        statusIcon().reset();
        abortController.abort();
        logger.debug("Completion request cancelled");
      });
    });
  }

  /**
   * Generates completions for the current document.
   * @param {vscode.TextDocument} document - The current document.
   * @param {vscode.Position} position - The current position in the document.
   * @param {AbortController} abortController - The abort controller for the operation.
   * @returns {Promise<vscode.InlineCompletionList>} A list of inline completions.
   */
  private async generateCompletions(
    document: vscode.TextDocument,
    position: vscode.Position,
    abortController: AbortController,
  ): Promise<vscode.InlineCompletionList> {
    logger.debug("Generating completions");
    const config = storage().get("completions.disabled.languages") || [];
    const languageId = document.languageId;
    if (config.includes(languageId)) {
      logger.debug(`Language ${languageId} is disabled for completions`);
      return new vscode.InlineCompletionList([]);
    }

    const provider =
      ModelProviderManager.getInstance().getProvider<"completion">(
        "Inline Completion",
      );
    if (!provider) {
      logger.warn("No model provider configured for `Inline Completion`");
      return new vscode.InlineCompletionList([]);
    }

    // Limit the number of tokens for prompt
    const promptTokenLimit =
      provider.config.contextWindow - this.maxOutputTokens;

    // Get the prefix for prompt
    let prefixRange = new vscode.Range(0, 0, position.line, 0);
    const pointerLinePrefix = new vscode.Range(prefixRange.end, position);
    if (document.getText(pointerLinePrefix).trim()) {
      prefixRange = prefixRange.with({ end: position });
    }
    const prefixText = document.getText(prefixRange);

    // Get the suffix for prompt
    const suffixStart = position.with({ character: 0 }).translate(1);
    const suffixEnd = new vscode.Position(document.lineCount + 1, 0);
    let suffixRange = new vscode.Range(suffixStart, suffixEnd);
    const pointerLineSuffix = new vscode.Range(
      position,
      document.lineAt(position).range.end,
    );
    const pointerLineSuffixText = document.getText(pointerLineSuffix);
    let suffixText = document.getText(suffixRange);
    if (/[a-zA-Z0-9]/.test(pointerLineSuffixText.trim())) {
      suffixRange = suffixRange.with({ start: position });
      suffixText = document.getText(suffixRange);
    }

    // Get the file header
    const language = getLanguageConfig(document.languageId);
    const start = language.comment.start;
    const end = language.comment.end ?? "";
    const headerLines = [
      `File Path: \`${document.uri.fsPath}\``,
      `Language ID: \`${language.markdown}\``,
    ];
    const header = headerLines
      .map((line) => `${start} ${line} ${end}`)
      .join("\n");

    // Get the tokens for the header, prefix, and suffix
    const headerTokens = await provider.encode(header + "\n".repeat(2));
    let suffixTokens = await provider.encode(suffixText.trimEnd());
    let prefixTokens = await provider.encode(prefixText.trimStart());

    // Truncate prefix or suffix if it exceeds the context window
    const maxPrefixLength =
      Math.floor(
        storage().workspace.get<number>(
          "flexpilot.completions.contextPrefixWeight",
        ) * promptTokenLimit,
      ) - headerTokens.length;
    const maxSuffixLength = promptTokenLimit - maxPrefixLength;
    if (
      prefixTokens.length > maxPrefixLength &&
      suffixTokens.length > maxSuffixLength
    ) {
      prefixTokens = prefixTokens.slice(-maxPrefixLength);
      suffixTokens = suffixTokens.slice(0, maxSuffixLength);
      logger.debug("Truncated both prefix and suffix tokens");
    } else if (
      prefixTokens.length > maxPrefixLength &&
      suffixTokens.length <= maxSuffixLength
    ) {
      const diff = suffixTokens.length - promptTokenLimit;
      prefixTokens = prefixTokens.slice(diff);
      logger.debug("Truncated prefix tokens");
    } else if (
      prefixTokens.length <= maxPrefixLength &&
      suffixTokens.length > maxSuffixLength
    ) {
      const diff = promptTokenLimit - prefixTokens.length;
      suffixTokens = suffixTokens.slice(0, diff);
      logger.debug("Truncated suffix tokens");
    }

    // Add header tokens before prefix
    prefixTokens = headerTokens.concat(prefixTokens);
    const prefix = prefixTokens.length
      ? await provider.decode(prefixTokens)
      : "";
    const suffix = suffixTokens.length
      ? await provider.decode(suffixTokens)
      : "";

    // Set multiple line generations only if it's a continued generation
    const stop = document.lineAt(position).text.trim()
      ? ["\n"]
      : ["\n".repeat(2)];
    // Stopper from suffix helps reduce code repetition from suffix
    const suffixLines = suffix
      .split("\n")
      .map((item) => item.trim())
      .filter((item) => item.trim());
    if (suffixLines.length > 1) {
      stop.push(suffixLines[0]);
    }

    // Invoke the completion model
    const response = await provider.invoke({
      maxTokens: this.maxOutputTokens,
      stop: stop,
      signal: abortController.signal,
      temperature: storage().workspace.get<number>(
        "flexpilot.completions.temperature",
      ),
      messages: {
        suffix: suffix.trimEnd(),
        prefix: prefix.trimStart(),
      },
    });

    if (!response) {
      logger.debug("No response from provider");
      return new vscode.InlineCompletionList([]);
    }

    // Get the range to insert the completion
    let insertRange = pointerLinePrefix;
    let insertText = response;
    // If the prefix is empty, set the insert range to the start of the line
    if (document.getText(insertRange).trim()) {
      insertText = document.getText(insertRange) + response;
    }
    if (pointerLineSuffixText && !pointerLineSuffixText.trim()) {
      // If the suffix contains only empty spaces or tabs
      insertText = insertText.trimEnd();
    } else if (!/[a-zA-Z0-9]/.test(pointerLineSuffixText.trim())) {
      // If the suffix contains non-alphanumeric characters only
      let partResponse = response;
      for (const char of pointerLineSuffixText) {
        const charIndex = partResponse.indexOf(char);
        if (!char.trim()) {
          // If empty space or tab character
          insertRange = insertRange.with({
            end: insertRange.end.translate(0, 1),
          });
        } else if (charIndex >= 0) {
          // If special character part of response
          partResponse = partResponse.slice(charIndex);
          insertRange = insertRange.with({
            end: insertRange.end.translate(0, 1),
          });
        } else {
          // Break if the character not in response
          break;
        }
      }
    }

    // Remove empty lines at the end of the string
    const lines = insertText.split("\n");
    while (lines.length > 0 && lines[lines.length - 1].trim() === "") {
      lines.pop();
    }
    insertText = lines.join("\n");

    // Store cache for the current completion
    const cachePrefix = document.getText(
      new vscode.Range(new vscode.Position(0, 0), insertRange.start),
    );
    const cacheSuffix = document.getText(
      new vscode.Range(
        insertRange.end,
        new vscode.Position(document.lineCount + 1, 0),
      ),
    );
    this.cachePrefix = cachePrefix + insertText;
    this.cacheSuffix = cacheSuffix;

    logger.debug("Completion generated successfully");
    return new vscode.InlineCompletionList([
      { insertText: insertText, range: insertRange },
    ]);
  }

  /**
   * Retrieves the cached completion items if available for the current position.
   * @param {vscode.TextDocument} document - The current document.
   * @param {vscode.Position} position - The current position in the document.
   * @returns {vscode.InlineCompletionList | undefined} Cached completion items or undefined.
   */
  private checkCompletionCache(
    document: vscode.TextDocument,
    position: vscode.Position,
  ): vscode.InlineCompletionList | undefined {
    logger.debug("Checking completion cache");
    const prefix = document.getText(
      new vscode.Range(new vscode.Position(0, 0), position),
    );
    const suffix = document.getText(
      new vscode.Range(
        position,
        new vscode.Position(document.lineCount + 1, 0),
      ),
    );
    if (
      this.cachePrefix.trim().length &&
      this.cachePrefix.startsWith(prefix) &&
      this.cacheSuffix.trim().length &&
      this.cacheSuffix === suffix
    ) {
      logger.debug("Cache hit");
      return new vscode.InlineCompletionList([
        {
          insertText: this.cachePrefix.slice(prefix.length),
          range: new vscode.Range(position, position),
        },
      ]);
    }
    logger.debug("Cache miss");
    return undefined;
  }
}

// Export the InlineCompletionProvider class
export default InlineCompletionProvider;
