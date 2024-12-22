import {
  CommentJSONValue,
  parse,
  stringify,
  type CommentObject,
} from "comment-json";
import * as path from "path";
import * as vscode from "vscode";
import { IPackageJson } from "./interfaces";
import { logger } from "./logger";
import { storage } from "./storage";
import { isFileExists } from "./utilities";

/**
 * Show notification to restart VS Code to apply changes
 */
const triggerVscodeRestart = async () => {
  // Get the current value of the titleBarStyle setting
  const existingValue = vscode.workspace
    .getConfiguration("window")
    .get("titleBarStyle");

  // Toggle the value of the titleBarStyle setting
  await vscode.workspace
    .getConfiguration("window")
    .update(
      "titleBarStyle",
      existingValue === "native" ? "custom" : "native",
      vscode.ConfigurationTarget.Global,
    );

  // Sleep for few milliseconds
  await new Promise((resolve) => setTimeout(resolve, 200));

  // Toggle the value back to the original value
  await vscode.workspace
    .getConfiguration("window")
    .update(
      "titleBarStyle",
      existingValue === "native" ? "native" : "custom",
      vscode.ConfigurationTarget.Global,
    );
};

/**
 * Checks if the proposed API is disabled in the current environment.
 */
const isProposedApiDisabled = async () => {
  try {
    await vscode.lm.fileIsIgnored(vscode.Uri.file("/"), {
      isCancellationRequested: false,
      onCancellationRequested: () => new vscode.EventEmitter(),
    });
    return false;
  } catch (error) {
    logger.error(error as Error);
    logger.error("Proposed API disabled for Flexpilot");
    return true;
  }
};

/**
 * Checks if GitHub Copilot is active in the current environment.
 */
const isGitHubCopilotActive = () => {
  // Get the extension by its identifier
  const extension = vscode.extensions.getExtension("GitHub.copilot");

  // Check if the extension is installed
  if (extension) {
    logger.info("GitHub Copilot is installed");
    return true;
  } else {
    logger.info("GitHub Copilot is not installed.");
    return false;
  }
};

/**
 * Checks if the package.json file is outdated and updates it.
 */
const isPackageJsonOutdated = async () => {
  // Get the path of the package.json file
  const packageJsonUri = vscode.Uri.joinPath(
    storage.getContext().extensionUri,
    "package.json",
  );
  logger.debug(`Package JSON Path: ${packageJsonUri}`);

  // Get the loaded package.json content
  const loadedPackageJson = storage.getContext().extension.packageJSON;

  // Check if the package.json file is outdated
  let isPackageJsonOutdated = false;

  // Parse the package.json content
  const packageJsonBuffer = await vscode.workspace.fs.readFile(packageJsonUri);
  const packageJson: IPackageJson = JSON.parse(
    Buffer.from(packageJsonBuffer).toString("utf8"),
  );

  if (loadedPackageJson.contributes.menus["scm/inputBox"] === undefined) {
    // Add the "scm/inputBox" menu to the "contributes" section
    packageJson.contributes.menus["scm/inputBox"] = [
      {
        when: "scmProvider == git",
        command: "flexpilot.git.generateCommitMessage",
      },
    ];

    // Set flag to update the package.json file
    isPackageJsonOutdated = true;
    logger.info("Successfully updated package with 'scm/inputBox'");
  }

  if (loadedPackageJson.enabledApiProposals === undefined) {
    // Add the proposed API to the "contributes" section
    packageJson.enabledApiProposals = packageJson.enabledApiProposalsOriginal;

    // Set flag to update the package.json file
    isPackageJsonOutdated = true;
    logger.info("Successfully updated package with 'enabledApiProposals'");
  }

  if (loadedPackageJson.contributes.languageModels === undefined) {
    // Add the "languageModels" section to the "contributes" section
    packageJson.contributes.languageModels = {
      vendor: "flexpilot",
    };

    // Set flag to update the package.json file
    isPackageJsonOutdated = true;
    logger.info("Successfully updated package with 'scm/inputBox'");
  }

  if (isPackageJsonOutdated) {
    // Write the changes to the package.json file
    await vscode.workspace.fs.writeFile(
      packageJsonUri,
      Buffer.from(JSON.stringify(packageJson, null, 2), "utf8"),
    );
  }

  // Return the flag indicating if the package.json file was updated
  return isPackageJsonOutdated;
};

/**
 * Checks if the argv.json file is outdated and updates it.
 */
const isArgvJsonOutdated = async () => {
  // Get the path of the argv.json file from storage
  let argvPath = storage.get("argv.path");
  let argvJsonOutdated = false;

  // Check if the argv.json file is outdated and trigger the update
  if (argvPath && !isFileExists(vscode.Uri.parse(argvPath))) {
    argvPath = undefined;
  }

  if (!argvPath) {
    // Open runtime arguments configuration
    await vscode.commands.executeCommand(
      "workbench.action.configureRuntimeArguments",
    );

    // Find the argv.json file in visible text editors
    const argvDocument = vscode.window.visibleTextEditors
      .map((item) => item.document)
      .filter((item) => path.basename(item.uri.fsPath) === "argv.json")
      .pop();

    // throw an error if the file is not found
    if (!argvDocument) {
      throw new Error("argv.json file not found in visible editors");
    }

    // Set the argv path in storage
    argvPath = argvDocument.uri.toString();
    await storage.set("argv.path", argvPath);

    // Make the argv file active and close it
    await vscode.window.showTextDocument(argvDocument);
    await vscode.commands.executeCommand("workbench.action.closeActiveEditor");
  }

  // Create the URI for the argv.json file
  const argvFileUri = vscode.Uri.parse(argvPath);

  // Get the extension ID
  const extensionId = storage.getContext().extension.id;
  logger.debug(`Extension ID: ${extensionId}`);

  // Parse the argv.json content
  const buffer = await vscode.workspace.fs.readFile(argvFileUri);
  const argvJson = parse(Buffer.from(buffer).toString("utf8")) as CommentObject;

  // Add the extension ID to the "enable-proposed-api" array
  if (Array.isArray(argvJson["enable-proposed-api"])) {
    if (!argvJson["enable-proposed-api"].includes(extensionId)) {
      argvJsonOutdated = true;
      argvJson["enable-proposed-api"].push(extensionId);
      logger.debug(`Added ${extensionId} to existing array`);
    } else {
      logger.debug(`${extensionId} already in array`);
    }
  } else if (typeof argvJson["enable-proposed-api"] === "string") {
    argvJsonOutdated = true;
    argvJson["enable-proposed-api"] = [
      argvJson["enable-proposed-api"],
      extensionId,
    ] as CommentJSONValue;
    logger.debug(`Created new array with ${extensionId}`);
  } else {
    argvJsonOutdated = true;
    argvJson["enable-proposed-api"] = [extensionId] as CommentJSONValue;
    logger.debug(`Created new array with ${extensionId}`);
  }

  // Add the extension ID to the "log-level" array
  const logLevel = `${extensionId}=debug`;
  if (Array.isArray(argvJson["log-level"])) {
    if (!argvJson["log-level"].includes(logLevel)) {
      argvJsonOutdated = true;
      argvJson["log-level"].push(logLevel);
      logger.debug(`Added ${logLevel} to existing array`);
    } else {
      logger.debug(`${logLevel} already in array`);
    }
  } else if (typeof argvJson["log-level"] === "string") {
    argvJsonOutdated = true;
    argvJson["log-level"] = [
      argvJson["log-level"],
      logLevel,
    ] as CommentJSONValue;
    logger.debug(`Created new array with ${logLevel}`);
  } else {
    argvJsonOutdated = true;
    argvJson["log-level"] = [logLevel] as CommentJSONValue;
    logger.debug(`Created new array with ${logLevel}`);
  }

  // If the argv.json file was updated, write the changes
  if (argvJsonOutdated) {
    await vscode.workspace.fs.writeFile(
      argvFileUri,
      Buffer.from(stringify(argvJson, null, 4), "utf8"),
    );
    logger.info("Successfully updated argv.json");
  }

  // Return the flag indicating if the argv.json file was updated
  return argvJsonOutdated;
};

/**
 * updates the runtime arguments configuration to enable proposed API, log level, etc ...
 */
export const updateRuntimeArguments = async () => {
  // Initialize the flag to require a restart
  let requireRestart = false;
  let restartMessage =
    "Flexpilot: Please restart VS Code to apply the latest updates";

  // Hide the chat setup if it is visible
  try {
    const currentOfferSetup = vscode.workspace
      .getConfiguration("chat.experimental")
      .get("offerSetup");
    if (currentOfferSetup != false) {
      vscode.workspace
        .getConfiguration("chat.experimental")
        .update("offerSetup", false, vscode.ConfigurationTarget.Global);
    }
  } catch (error) {
    logger.warn(`Chat setup not found: ${String(error)}`);
  }

  // Check if the argv.json file is outdated
  if (await isArgvJsonOutdated()) {
    logger.warn("Updated runtime arguments, restart required");
    requireRestart = true;
  }

  // Check if the package.json file is outdated
  if (await isPackageJsonOutdated()) {
    logger.warn("Updated package.json, restart required");
    requireRestart = true;
  }

  // Check if the proposed API is disabled
  if (await isProposedApiDisabled()) {
    logger.warn("Proposed API is disabled, restart required");
    requireRestart = true;
  }

  // Check if GitHub Copilot is active
  if (isGitHubCopilotActive()) {
    logger.warn("GitHub Copilot is active, restart required");
    restartMessage =
      "Flexpilot: To ensure Flexpilot functions correctly, kindly disable `GitHub Copilot` and Restart";
  }

  // Notify the user about the required restart
  if (requireRestart) {
    // Show a notification to restart VS Code
    vscode.window
      .showInformationMessage(restartMessage, "Restart", "View Logs")
      .then((selection) => {
        if (selection === "Restart") {
          triggerVscodeRestart();
        } else if (selection === "View Logs") {
          logger.showOutputChannel();
        }
      });

    // Throw an error to stop the execution
    throw new Error("Flexpilot: VS Code restart required");
  }

  // Log the successful activation
  logger.info("Successfully updated runtime arguments");
};
