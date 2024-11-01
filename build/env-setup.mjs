// import necessary modules
import * as fs from "fs";
import { Octokit } from "octokit";
import * as path from "path";
import * as semver from "semver";

// Create a new instance of the Octokit client
const github = new Octokit();

// Read the package.json file to get the version number
let packageJson = JSON.parse(await fs.readFileSync("package.json", "utf-8"));

// Get the vscode release version
const version = semver.parse(packageJson.version);

// Check if the branch exists in the vscode repository
let vscodeBranch = `release/${version.major}.${version.minor}`;
try {
  await github.rest.repos.getBranch({
    owner: "microsoft",
    repo: "vscode",
    branch: vscodeBranch,
  });
  // Log the branch being used
  console.log(`[INFO] Fetching from branch: ${vscodeBranch}`);
} catch {
  // If the branch does not exist, use the main branch
  console.warn(`[WARN] Branch does not exist`);
  vscodeBranch = "main";
  console.warn(`[WARN] Using main branch instead`);
}

console.log("Fetching from branch:", vscodeBranch);

// Get the list of proposed APIs from the vscode repository
const vscodeDtsFiles = await github.rest.repos.getContent({
  owner: "microsoft",
  repo: "vscode",
  ref: vscodeBranch,
  path: "src/vscode-dts",
});

// Filter out the proposed APIs from the list of files
const proposedApis = vscodeDtsFiles.data
  .map((item) => item.name)
  .filter((item) => item.startsWith("vscode.proposed."))
  .filter((item) => item.endsWith(".d.ts"))
  .map((item) => item.replace("vscode.proposed.", ""))
  .map((item) => item.replace(".d.ts", ""));

// Update the package.json file
packageJson.enabledApiProposalsOriginal = proposedApis;
packageJson.engines = {};
packageJson.engines.vscode = `${version.major}.${version.minor}.x`;

// Write the updated package.json file
await fs.writeFileSync("package.json", JSON.stringify(packageJson, null, 2));

// Get the list of git APIs from the vscode repository
const gitDtsFiles = await github.rest.repos.getContent({
  owner: "microsoft",
  repo: "vscode",
  ref: vscodeBranch,
  path: "extensions/git/src/api",
});

// Filter only ts declaration files
const declarationFiles = vscodeDtsFiles.data
  .filter((item) => item.name.endsWith(".d.ts"))
  .concat(gitDtsFiles.data.filter((item) => item.name.endsWith(".d.ts")));

// Create the types directory if it doesn't exist
if (!fs.existsSync("./types")) {
  fs.mkdirSync("./types", { recursive: true });
}

// Download the .d.ts files from the vscode repository
for (const element of declarationFiles) {
  const item = element;
  const fileName = path.basename(item.download_url);
  const filePath = path.join("./types", fileName);
  const response = await fetch(item.download_url);
  const buffer = await response.arrayBuffer();
  fs.writeFileSync(filePath, Buffer.from(buffer));
}
