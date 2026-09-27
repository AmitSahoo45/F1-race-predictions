import 'server-only';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import Ajv2020 from 'ajv/dist/2020.js';
import addFormats from 'ajv-formats';
import type { SiteData } from '@/generated/site-data';

let cached: SiteData | undefined;

export function getSiteData(): SiteData {
  if (cached) return cached;
  const schemaPath = resolve(process.cwd(), '../schemas/site.schema.json');
  if (process.env.APEX_SITE_DATA_PATH && process.env.APEX_ALLOW_TEST_FIXTURE !== '1') throw new Error('Test fixture override requires APEX_ALLOW_TEST_FIXTURE=1');
  const dataPath = process.env.APEX_SITE_DATA_PATH ? resolve(process.cwd(), process.env.APEX_SITE_DATA_PATH) : resolve(process.cwd(), '../site-data/site.json');
  const schema = JSON.parse(readFileSync(schemaPath, 'utf8'));
  const site = JSON.parse(readFileSync(dataPath, 'utf8'));
  const ajv = new Ajv2020({ allErrors: true, strict: false });
  addFormats(ajv);
  const validate = ajv.compile<SiteData>(schema);
  if (!validate(site)) {
    throw new Error(`Invalid site-data/site.json: ${ajv.errorsText(validate.errors, { separator: '\n' })}`);
  }
  cached = site;
  return site;
}
