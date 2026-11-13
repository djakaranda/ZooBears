// Finalize the editable draft exported from the live presentation session.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const skill='C:/Users/Макар/.codex/plugins/cache/openai-primary-runtime/presentations/26.1112.11217/skills/presentations';
const python='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES??='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
await fs.mkdir(path.join(root,'output/presentation'),{recursive:true});
const result=await finalizePresentation({
  workspaceDir:root,
  candidatePath:path.join(root,'tmp/slides/candidate-v3.pptx'),
  finalPath:path.join(root,'output/presentation/ZooBears_progress.pptx'),
  pythonExecutable:python,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
              '--require-native-table-slide','4','--require-native-table-slide','6'],
  requiredNativeTableOwnerSlides:[4,6],requiredNativeChartOwnerSlides:[7],
  materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
  receiptPath:path.join(root,'tmp/slides/presentation.validation.json'),
});
console.log(JSON.stringify(result,null,2));
