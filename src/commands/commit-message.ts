import { generateText } from "ai";
import MarkdownIt from "markdown-it";
import * as vscode from "vscode";
import { GitExtension } from "../../types/git";
import { logger } from "../logger";
import { CommitMessagePrompt } from "../prompts/commands/commit-message";
import { ModelProviderManager } from "../providers";
import { storage } from "../storage";

/**
 * CommitMessageCommand class manages the commit message generation functionality.
 * It implements the Singleton pattern to ensure a single instance across the application.
 */
export class CommitMessageCommand extends vscode.Disposable {
  private static instance: CommitMessageCommand;
  private readonly disposable: vscode.Disposable;

  /**
   * Private constructor to prevent direct instantiation.
   * Registers the command and initializes the disposable.
   */
  private constructor() {
    // Call the parent constructor
    super(() => this.disposable.dispose());

    // Register the command
    this.disposable = vscode.commands.registerCommand(
      "flexpilot.git.generateCommitMessage",
      this.handler.bind(this),
    );
    logger.info("CommitMessageCommand instance created");
  }

  /**
   * Gets the singleton instance of CommitMessageCommand.
   * @returns {CommitMessageCommand} The singleton instance.
   */
  public static register() {
    if (!CommitMessageCommand.instance) {
      CommitMessageCommand.instance = new CommitMessageCommand();
      storage().context.subscriptions.push(CommitMessageCommand.instance);
      logger.debug("New CommitMessageCommand instance created");
    }
  }

  /**
   * Handles the commit message generation process.
   * @param {vscode.Uri} repositoryUri - The URI of the Git repository.
   * @param {unknown} _ISCMInputValueProviderContext - SCM input value provider context (unused).
   * @param {vscode.CancellationToken} token - Cancellation token for the operation.
   */
  public async handler(
    repositoryUri: vscode.Uri,
    _ISCMInputValueProviderContext: unknown,
    token: vscode.CancellationToken,
  ): Promise<void> {
    try {
      await vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.SourceControl,
        },
        async () => {
          logger.info("Starting commit message generation");

          // Create an AbortController to handle cancellation
          const abortController = new AbortController();
          token.onCancellationRequested(() => {
            abortController.abort();
            logger.info("Commit message generation cancelled by user");
          });

          // Get the model provider for the commit message
          const provider =
            ModelProviderManager.getInstance().getProvider<"chat">(
              "Commit Message",
            );
          if (!provider) {
            logger.notifyError("Model not configured for `Commit Message`");
            return;
          }

          // Get the Git repository
          const gitExtension =
            vscode.extensions.getExtension<GitExtension>("vscode.git");
          if (!gitExtension) {
            throw new Error("Git extension is not installed or enabled.");
          }

          // Get the repository from the Git extension
          const repository = gitExtension.exports
            .getAPI(1)
            .getRepository(repositoryUri);
          if (!repository) {
            throw new Error(`Git Repository not found at ${repositoryUri}`);
          }

          // Prepare prompts for the chat
          const messages = await CommitMessagePrompt.build(repository);
          if (!messages) {
            logger.info("No prompt messages generated, skipping");
            return;
          }

          // Generate the commit message
          const response = await generateText({
            model: await provider.model(),
            messages: messages,
            abortSignal: abortController.signal,
            stopSequences: [],
            temperature: storage().workspace.get<number>(
              "flexpilot.gitCommitMessage.temperature",
            ),
          });

          // Log the model response
          logger.debug(`Model Response: ${response.text}`);

          // Parse the response for the commit message
          const parsedTokens = new MarkdownIt()
            .parse(response.text, {})
            .filter((token) => token.type === "fence" && token.tag === "code");
          if (
            parsedTokens.length < 1 ||
            parsedTokens[0].content.trim().length < 1
          ) {
            throw new Error("Model did not return a valid commit message");
          }

          const commitMessage = parsedTokens[0].content.trim();
          repository.inputBox.value = commitMessage;

          // Log the completion of the commit message generation
          logger.info("Commit message generation completed successfully");
        },
      );
    } catch (error) {
      logger.error(error as Error);
      logger.notifyError("Error processing `Commit Message` request");
    }
  }
}
