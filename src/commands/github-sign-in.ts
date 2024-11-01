import * as vscode from "vscode";
import { logger } from "../logger";
import { storage } from "../storage";

/**
 * GithubSignInCommand class manages the GitHub sign-in functionality.
 * It implements the Singleton pattern to ensure a single instance across the application.
 */
export class GithubSignInCommand extends vscode.Disposable {
  private static instance: GithubSignInCommand;
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
      "flexpilot.github.signin",
      this.handler.bind(this),
    );
    logger.info("GithubSignInCommand instance created");
  }

  /**
   * Gets the singleton instance of GithubSignInCommand.
   * @returns {GithubSignInCommand} The singleton instance.
   */
  public static register() {
    if (!GithubSignInCommand.instance) {
      GithubSignInCommand.instance = new GithubSignInCommand();
      storage().context.subscriptions.push(GithubSignInCommand.instance);
      logger.debug("New GithubSignInCommand instance created");
    }
  }

  /**
   * Handles the GitHub sign-in process.
   * Prompts for GitHub star support if not previously set, then initiates the sign-in.
   */
  public async handler(): Promise<void> {
    try {
      logger.info("Handling `GithubSignInCommand`");
      // Check if the user has already accepted GitHub support
      const githubSupportStatus = storage().get("github.support");

      if (!githubSupportStatus) {
        const shouldSupport = await this.promptForGithubSupport();
        logger.info(
          `User opted to ${
            shouldSupport ? "support" : "not support"
          } the project with a GitHub star`,
        );
        await storage().set("github.support", shouldSupport);
      }

      await vscode.authentication.getSession("github", ["public_repo"], {
        createIfNone: true,
      });

      // Set the context to indicate successful sign-in for walkthroughs
      await vscode.commands.executeCommand(
        "setContext",
        "flexpilot:walkthroughSignin",
        true,
      );

      logger.notifyInfo("Successfully signed in with GitHub");
    } catch (error) {
      logger.error(error as Error);
      logger.notifyError("Error in `Github Sign In` command");
    }
  }

  /**
   * Prompts the user to support the project with a GitHub star.
   * @returns {Promise<boolean>} True if the user agrees to support, false otherwise.
   */
  private async promptForGithubSupport(): Promise<boolean> {
    const selectedOption = await vscode.window.showInformationMessage(
      "Flexpilot: Support Us!",
      {
        modal: true,
        detail:
          "Help our open-source project stay alive. We'll auto-star on GitHub when you sign in. No extra steps!",
      },
      "Proceed to Login",
      "No, I don't support",
    );
    return !selectedOption || selectedOption === "Proceed to Login";
  }
}
