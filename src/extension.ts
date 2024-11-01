// Import logger first to make sure it is initialized before other imports.
import { logger } from "./logger";

// Other imports
import * as vscode from "vscode";
import { CommitMessageCommand } from "./commands/commit-message";
import { ConfigureModelCommand } from "./commands/configure-model";
import { GithubSignInCommand } from "./commands/github-sign-in";
import { StatusIconMenuCommand } from "./commands/status-icon-menu";
import { ProxyModelProvider } from "./models";
import { SessionManager } from "./session";
import { updateRuntimeArguments } from "./startup";
import { statusIcon } from "./status-icon";
import { StorageManager } from "./storage";
import { setContext } from "./utilities";
import { VariablesManager } from "./variables";

/**
 * Activates the extension.
 */
export async function activate(context: vscode.ExtensionContext) {
  setContext("isLoaded", false);
  setContext("isError", false);
  setContext("isLoggedIn", false);
  try {
    // Initialize the storage manager
    StorageManager.createInstance(context);

    // Register the logger with the context
    context.subscriptions.push(logger);

    // Update the runtime arguments
    await updateRuntimeArguments();

    // Register the variables manager
    VariablesManager.register();

    // Register the proxy model
    ProxyModelProvider.register();

    // Register the commands
    StatusIconMenuCommand.register();
    CommitMessageCommand.register();
    GithubSignInCommand.register();
    ConfigureModelCommand.register();

    // Handle the session change
    SessionManager.register();

    // Update the status bar icon
    statusIcon().updateStatusBarIcon();

    // Set the loaded context
    setContext("isLoaded", true);
  } catch (error) {
    // Set the error context
    setContext("isError", true);

    // Log the error to the output channel
    logger.error(error as Error);
  }
}
