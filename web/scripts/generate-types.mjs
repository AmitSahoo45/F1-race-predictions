import { compileFromFile } from 'json-schema-to-typescript';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

for (const [schemaName, outputName] of [['site', 'site-data'], ['telemetry', 'telemetry']]) {
  const schema = fileURLToPath(new URL(`../../schemas/${schemaName}.schema.json`, import.meta.url));
  const output = fileURLToPath(new URL(`../src/generated/${outputName}.ts`, import.meta.url));
  const types = await compileFromFile(schema, { bannerComment: `// Generated from schemas/${schemaName}.schema.json. Do not edit by hand.\n`, additionalProperties: false });
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, types);
  console.log(`Generated src/generated/${outputName}.ts`);
}
