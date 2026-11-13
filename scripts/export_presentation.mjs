// Export the editable presentation source with the bundled Codex Artifact Tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const modules=process.env.RUNTIME_NODE_MODULES??'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules';
const require=createRequire(path.join(modules,'package.json'));
const {Presentation,PresentationFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const presentation=Presentation.load(JSON.parse(await fs.readFile(path.join(root,'reports/presentation_source.json'),'utf8')));
await fs.mkdir(path.join(root,'tmp/slides'),{recursive:true});
await (await PresentationFile.exportPptx(presentation)).save(path.join(root,'tmp/slides/candidate-v3.pptx'));
