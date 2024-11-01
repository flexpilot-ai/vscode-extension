import { CoreMessage } from "ai";
import * as vscode from "vscode";
import { Repository } from "../../../types/git";
import { logger } from "../../logger";
import { Code, jsxToCoreMessage, Message } from "../jsx-utilities";

/**
 * CommitMessagePrompt class handles the generation of prompts for commit message suggestions.
 * It manages the creation of system, code diff, repository commits, and user prompts.
 */
export class CommitMessagePrompt {
  /**
   * Generates the system prompt for commit message generation.
   * @returns {CoreMessage} The system prompt as a CoreMessage.
   */
  private static getSystemPrompt(): CoreMessage {
    logger.debug("Generating system prompt for commit message");
    return jsxToCoreMessage(
      <Message role="system">
        <p>
          You are an AI programming assistant, helping a software developer to
          come up with the best git commit message for their code changes. You
          excel in interpreting the purpose behind code changes to craft
          succinct, clear commit messages that adhere to the repository's
          guidelines.
        </p>
        <h1>Examples of commit messages:</h1>
        <ul>
          <li>
            <Code language="text">
              feat: improve page load with lazy loading for images
            </Code>
          </li>
          <li>
            <Code language="text">
              Fix bug preventing submitting the signup form
            </Code>
          </li>
          <li>
            <Code language="text">
              chore: update npm dependency to latest stable version
            </Code>
          </li>
          <li>
            <Code language="text">
              Update landing page banner color per client request
            </Code>
          </li>
        </ul>
        <ul>
          <li>
            First, think step-by-step, Analyze the CODE CHANGES thoroughly to
            understand what's been modified.
          </li>
          <li>
            Identify the purpose of the changes to answer the "why" for the
            commit messages, also considering the optionally provided RECENT
            USER COMMITS.
          </li>
          <li>
            Review the provided RECENT REPOSITORY COMMITS to identify
            established commit message conventions. Focus on the format and
            style, ignoring commit-specific details like refs, tags, and
            authors.
          </li>
          <li>
            Generate a thoughtful and succinct commit message for the given CODE
            CHANGES. It MUST follow the established writing conventions.
          </li>
          <li>
            Remove any meta information like issue references, tags, or author
            names from the commit message. The developer will add them.
          </li>
          <li>
            Now only show your message, wrapped with a single markdown ```text
            codeblock! Do not provide any explanations or details.
          </li>
        </ul>
      </Message>,
    );
  }

  /**
   * Generates the code diff prompt.
   * @param {string} diff - The git diff string.
   * @returns {CoreMessage} The code diff prompt as a CoreMessage.
   */
  private static getCodeDiffPrompt(diff: string): CoreMessage {
    logger.debug("Generating code diff prompt");
    return jsxToCoreMessage(
      <Message role="user">
        <h1>CODE CHANGES:</h1>
        <Code language="git-diff">{diff}</Code>
      </Message>,
    );
  }

  /**
   * Generates the repository commits prompt.
   * @param {Repository} repository - The git repository.
   * @returns {Promise<CoreMessage | undefined>} A promise that resolves to the repository commits prompt as a CoreMessage, or undefined if no commits are found.
   */
  private static async getRepositoryCommitsPrompt(
    repository: Repository,
  ): Promise<CoreMessage | undefined> {
    logger.debug("Generating repository commits prompt");
    const recentCommits = await repository.log({ maxEntries: 10 });
    if (!recentCommits.length) {
      logger.debug("No recent commits found");
      return undefined;
    }
    return jsxToCoreMessage(
      <Message role="user">
        <h1>RECENT REPOSITORY COMMITS:</h1>
        <ul>
          {recentCommits.map((log) => (
            <li key={log.message}>
              <Code language="text">{log.message}</Code>
            </li>
          ))}
        </ul>
      </Message>,
    );
  }

  /**
   * Generates the author commits prompt.
   * @param {Repository} repository - The git repository.
   * @returns {Promise<CoreMessage | undefined>} A promise that resolves to the author commits prompt as a CoreMessage, or undefined if no commits are found.
   */
  private static async getAuthorCommitsPrompt(
    repository: Repository,
  ): Promise<CoreMessage | undefined> {
    logger.debug("Generating author commits prompt");
    const authorName =
      (await repository.getConfig("user.name")) ??
      (await repository.getGlobalConfig("user.name"));
    const authorCommits = await repository.log({
      maxEntries: 10,
      author: authorName,
    });
    if (!authorCommits.length) {
      logger.debug("No author commits found");
      return undefined;
    }
    return jsxToCoreMessage(
      <Message role="user">
        <h1>RECENT AUTHOR COMMITS:</h1>
        <ul>
          {authorCommits.map((log) => (
            <li key={log.message}>
              <Code language="text">{log.message}</Code>
            </li>
          ))}
        </ul>
      </Message>,
    );
  }

  /**
   * Generates the user prompt.
   * @returns {CoreMessage} The user prompt as a CoreMessage.
   */
  private static getUserPrompt(): CoreMessage {
    logger.debug("Generating user prompt");
    return jsxToCoreMessage(
      <Message role="user">
        <p>
          Remember to ONLY return a single markdown ```text code block with the
          suggested commit message. NO OTHER PROSE! If you write more than the
          commit message, your commit message gets lost.
        </p>
        <h1>Example:</h1>
        <Code language="test">commit message goes here</Code>
      </Message>,
    );
  }

  /**
   * Handles the case when no diff is found.
   * @param {Repository} repository - The git repository.
   */
  private static handleNoDiff(repository: Repository): void {
    logger.warn("No staged changes found to commit");
    vscode.window
      .showErrorMessage(
        "No staged changes found to commit. Please stage changes and try again.",
        "Stage All Changes",
      )
      .then((selection) => {
        if (selection === "Stage All Changes") {
          logger.info("User chose to stage all changes");
          vscode.commands.executeCommand("git.stageAll", repository);
        }
      });
  }

  /**
   * Builds the complete set of prompts for commit message generation.
   * @param {Repository} repository - The git repository.
   * @returns {Promise<CoreMessage[] | undefined>} A promise that resolves to an array of CoreMessages or undefined if no diff is found.
   */
  public static async build(
    repository: Repository,
  ): Promise<CoreMessage[] | undefined> {
    logger.info("Building commit message prompts");

    const diff =
      (await repository.diff(true)) || (await repository.diff(false));
    if (!diff) {
      this.handleNoDiff(repository);
      return;
    }
    const messages: CoreMessage[] = [
      this.getSystemPrompt(),
      this.getCodeDiffPrompt(diff),
    ];
    const repoCommits = await this.getRepositoryCommitsPrompt(repository);
    if (repoCommits) {
      messages.push(repoCommits);
    }
    const authorCommits = await this.getAuthorCommitsPrompt(repository);
    if (authorCommits) {
      messages.push(authorCommits);
    }
    messages.push(this.getUserPrompt());
    logger.info("Successfully built commit message prompts");
    return messages;
  }
}
