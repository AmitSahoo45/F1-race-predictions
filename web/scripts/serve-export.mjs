import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { brotliCompressSync, constants } from 'node:zlib';

const out = resolve(fileURLToPath(new URL('../out/', import.meta.url)));
const prefix = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
const port = Number(process.env.PORT ?? 3100);
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.json': 'application/json', '.woff2': 'font/woff2', '.png': 'image/png', '.ico': 'image/x-icon' };

createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url ?? '/', 'http://localhost').pathname);
    if (prefix && pathname !== prefix && !pathname.startsWith(`${prefix}/`)) { response.writeHead(404).end(); return; }
    const route = prefix ? pathname.slice(prefix.length) || '/' : pathname;
    let path = resolve(out, `.${route}`);
    if (path !== out && !path.startsWith(`${out}${sep}`)) { response.writeHead(403).end(); return; }
    if ((await stat(path)).isDirectory()) path = resolve(path, 'index.html');
    const body = await readFile(path);
    const canCompress = /\.(html|js|css|svg|json)$/.test(path) && request.headers['accept-encoding']?.includes('br');
    const payload = canCompress ? brotliCompressSync(body, { params: { [constants.BROTLI_PARAM_QUALITY]: 4 } }) : body;
    response.writeHead(200, { 'Content-Type': mime[extname(path)] ?? 'application/octet-stream', 'Cache-Control': 'no-cache', 'Content-Encoding': canCompress ? 'br' : 'identity', 'Content-Length': payload.length, Vary: 'Accept-Encoding' });
    response.end(payload);
  } catch { response.writeHead(404).end('Not found'); }
}).listen(port, '127.0.0.1', () => console.log(`Serving static export at http://127.0.0.1:${port}${prefix}/`));
