import { Tokenizer } from "@flexpilot-ai/tokenizers";
import assert from "assert";
import axios from "axios";
import { createHash } from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import { storage } from "./storage";
import { getCompletionModelMetadata } from "./utilities";

export class Tokenizers {
  /**
   * Get the tokenizer metadata for the given model.
   * @param {string} model - The name of the model.
   * @returns {Object} The tokenizer metadata.
   */
  private static metadata(model: string) {
    // Get the configuration for the model
    const metadata = getCompletionModelMetadata(model);

    // Check if the model configuration exists
    if (!metadata) {
      throw new Error("No tokenizer URL found for model");
    }

    // Prepare the tokenizer file path
    const basePath = storage().context.globalStorageUri.fsPath;
    let fileId = metadata.tokenizerUrl.split("/").pop();
    const tokenizersFolder = path.join(basePath, "tokenizers");
    if (!fileId) {
      fileId = createHash("sha512").update(model).digest("hex");
    }
    const jsonPath = path.join(tokenizersFolder, fileId);

    // Check if the tokenizer folder exists in storage
    if (!fs.existsSync(tokenizersFolder)) {
      fs.mkdirSync(tokenizersFolder, { recursive: true });
    }

    // Return the metadata and the path to the tokenizer file
    return { metadata, jsonPath };
  }

  /**
   * Get the tokenizer for the given model.
   * @param {string} model - The name of the model.
   * @returns {Tokenizer} The tokenizer object.
   */
  public static get(model: string): Tokenizer {
    const { jsonPath } = this.metadata(model);
    const data = fs.readFileSync(jsonPath);
    const byteArray = Array.from(new Uint8Array(data));
    return new Tokenizer(byteArray);
  }

  /**
   * Download the tokenizer for the given model.
   * @param {string} model - The name of the model.
   * @returns {Promise<Tokenizer>} The tokenizer object.
   */
  public static async download(model: string): Promise<Tokenizer> {
    const { metadata, jsonPath } = this.metadata(model);
    const response = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: "Flexpilot",
        cancellable: false,
      },
      async (progress) => {
        progress.report({
          message: "Downloading tokenizer.json",
        });
        return await axios.get(metadata.tokenizerUrl, {
          responseType: "arraybuffer",
        });
      },
    );
    const byteArray = Array.from(new Uint8Array(response.data));
    const tokenizer = new Tokenizer(byteArray);
    assert(tokenizer.encode("test string", false).length > 0);
    fs.writeFileSync(jsonPath, response.data, "utf8");
    return tokenizer;
  }
}
