import axios, { AxiosError } from "axios";
import * as vscode from "vscode";

/**
 * LoggerSingleton class provides a centralized logging mechanism for the Flexpilot VS Code extension.
 * It implements the Singleton pattern to ensure a single instance of the logger across the application.
 */
export class LoggerSingleton extends vscode.Disposable {
  private readonly disposable: vscode.Disposable;
  private static instance: LoggerSingleton;
  private readonly outputChannel: vscode.LogOutputChannel;

  /**
   * Creates a new instance of LoggerSingleton.
   */
  private constructor() {
    // Call the parent constructor
    super(() => {
      // Dispose the command registration
      this.disposable.dispose();

      // Dispose the output channel
      this.outputChannel.dispose();
    });

    // Create the output channel
    this.outputChannel = vscode.window.createOutputChannel("Flexpilot", {
      log: true,
    });

    // Register the command to view logs
    this.disposable = vscode.commands.registerCommand(
      "flexpilot.viewLogs",
      () => this.outputChannel.show(),
    );
  }

  /**
   * Returns the singleton instance of LoggerSingleton.
   * @returns {LoggerSingleton} The singleton instance.
   */
  public static getInstance(): LoggerSingleton {
    if (!LoggerSingleton.instance) {
      LoggerSingleton.instance = new LoggerSingleton();
    }
    return LoggerSingleton.instance;
  }

  /**
   * Logs an informational message to the output channel.
   */
  public info(message: string, ...args: unknown[]): void {
    this.outputChannel.info(message, ...args);
  }

  /**
   * Logs an informational message and shows a notification.
   */
  public notifyInfo(message: string, ...args: unknown[]): void {
    this.outputChannel.info(message, ...args);

    // Show information notification
    vscode.window
      .showInformationMessage(message, "Open Docs")
      .then(async (selection) => {
        if (selection === "Open Docs") {
          vscode.env.openExternal(
            vscode.Uri.parse("https://docs.flexpilot.ai/"),
          );
        }
      });
  }

  /**
   * Logs a warning message to the output channel.
   */
  public warn(message: string, ...args: unknown[]): void {
    this.outputChannel.warn(message, ...args);
  }

  /**
   * Logs a debug message to the output channel.
   */
  public debug(message: string, ...args: unknown[]): void {
    this.outputChannel.debug(message, ...args);
  }

  /**
   * Logs an error message to the output channel.
   */
  public error(error: string | Error, ...args: unknown[]): void {
    this.outputChannel.error(error, ...args);
  }

  /**
   * Logs an error message and shows a notification.
   */
  public notifyError(message: string): void {
    this.outputChannel.error(message);

    // Show error notification
    vscode.window
      .showErrorMessage(message, "View Details")
      .then((selection) => {
        if (selection === "View Details") {
          this.outputChannel.show();
        }
      });
  }
}

// Export a singleton instance of the logger
export const logger = LoggerSingleton.getInstance();

// Axios request interceptor
axios.interceptors.request.use(
  (config) => {
    // Log the request details and return the config
    logger.debug("Axios Request:", {
      headers: config.headers,
      url: config.url,
      method: config.method,
      data: config.data,
      baseURL: config.baseURL,
    });
    return config;
  },
  (error: AxiosError) => {
    // Log the request details and reject the promise
    logger.error("Axios Request:", error);
    return Promise.reject(error);
  },
);

// Axios response interceptor
axios.interceptors.response.use(
  (response) => {
    // Log the response details and return the response
    logger.debug("Axios Response:", {
      status: response.status,
      data: response.data,
      headers: response.headers,
    });
    return response;
  },
  (error: AxiosError) => {
    if (error.response) {
      // The request was made and the server responded with a status code
      logger.error("Axios Error:", {
        status: error.response.status,
        data: error.response.data,
        headers: error.response.headers,
      });
    } else if (error.request) {
      // No response received
      logger.error("Axios Error (No Response):", error.request);
    } else {
      // Something happened in setting up the request
      logger.error("Axios Error:", error.message);
    }
    return Promise.reject(error);
  },
);
