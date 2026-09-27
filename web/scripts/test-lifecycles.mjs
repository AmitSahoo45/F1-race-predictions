import { spawnSync } from 'node:child_process';

const corepack = process.platform === 'win32' ? 'corepack.cmd' : 'corepack';
const fixtureEnv = { ...process.env, APEX_ALLOW_TEST_FIXTURE: '1', APEX_SITE_DATA_PATH: 'tests/.generated/lifecycle-site.json' };
const normalEnv = { ...process.env };
delete normalEnv.APEX_ALLOW_TEST_FIXTURE;
delete normalEnv.APEX_SITE_DATA_PATH;

function run(command, args, env) {
  const result = spawnSync(command, args, { env, stdio: 'inherit', shell: process.platform === 'win32' && command === corepack });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} ${args.join(' ')} exited ${result.status ?? result.signal}`);
}

let failure;
try {
  run(process.execPath, ['scripts/make-lifecycle-fixture.mjs'], fixtureEnv);
  run(corepack, ['pnpm', 'build'], fixtureEnv);
  run(corepack, ['pnpm', 'verify:export'], fixtureEnv);
  const args = ['pnpm', 'test:e2e'];
  if (process.env.APEX_UPDATE_SNAPSHOTS === '1') args.push('--update-snapshots');
  run(corepack, args, fixtureEnv);
} catch (error) {
  failure = error;
} finally {
  try {
    run(corepack, ['pnpm', 'build'], normalEnv);
    run(corepack, ['pnpm', 'verify:export'], normalEnv);
  } catch (error) {
    failure ??= error;
  }
}
if (failure) throw failure;
