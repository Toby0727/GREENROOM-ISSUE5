import { promises as fs } from 'fs';
import path from 'path';

const STORE_DIR_CANDIDATES = [
  path.join(process.cwd(), '.data'),
  '/tmp'
];

const STORE_FILE_NAME = 'greenroom-comments.json';
const KV_HASH_KEY = 'greenroom:comments';

function isCommentRecord(record) {
  return Boolean(
    record &&
    typeof record.id === 'string' && record.id &&
    typeof record.text === 'string' &&
    typeof record.user === 'string' && record.user &&
    typeof record.device === 'string' && record.device &&
    typeof record.font === 'string' && record.font &&
    typeof (record.fontSize || 'md') === 'string' &&
    Number.isFinite(Number(record.ts))
  );
}

function normalizeComment(record) {
  if (!isCommentRecord(record)) return null;

  return {
    id: String(record.id),
    text: String(record.text),
    user: String(record.user),
    device: String(record.device),
    font: String(record.font),
    fontSize: String(record.fontSize || 'md'),
    ts: Number(record.ts)
  };
}

function canUseKv() {
  return Boolean(process.env.KV_REST_API_URL && process.env.KV_REST_API_TOKEN);
}

async function kvRequest(command, ...args) {
  const baseUrl = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;

  if (!baseUrl || !token) {
    throw new Error('Vercel KV is not configured');
  }

  const response = await fetch(`${baseUrl}/${[command, ...args].map(part => encodeURIComponent(String(part))).join('/')}`, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${token}`
    }
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `KV ${command} request failed`);
  }

  return payload.result;
}

async function readCommentsFromKv() {
  const result = await kvRequest('hgetall', KV_HASH_KEY);
  if (!Array.isArray(result) || result.length === 0) return [];

  const comments = [];
  for (let index = 1; index < result.length; index += 2) {
    try {
      const parsed = JSON.parse(result[index]);
      const normalized = normalizeComment(parsed);
      if (normalized) comments.push(normalized);
    } catch {}
  }

  return comments.sort((left, right) => left.ts - right.ts);
}

async function writeCommentsToKv(comments) {
  const normalized = Array.isArray(comments)
    ? comments.map(normalizeComment).filter(Boolean).sort((left, right) => left.ts - right.ts)
    : [];

  await kvRequest('del', KV_HASH_KEY);

  for (const comment of normalized) {
    await kvRequest('hset', KV_HASH_KEY, comment.id, JSON.stringify(comment));
  }

  return normalized;
}

async function appendCommentToKv(comment) {
  const normalized = normalizeComment(comment);
  if (!normalized) {
    throw new Error('Invalid comment payload');
  }

  await kvRequest('hset', KV_HASH_KEY, normalized.id, JSON.stringify(normalized));
  return readCommentsFromKv();
}

async function clearCommentsFromKv() {
  await kvRequest('del', KV_HASH_KEY);
  return [];
}

async function getReadableStorePath() {
  const writablePath = await getWritableStorePath();
  const candidatePaths = [
    writablePath,
    ...STORE_DIR_CANDIDATES.map(dirPath => path.join(dirPath, STORE_FILE_NAME))
  ].filter((filePath, index, paths) => paths.indexOf(filePath) === index);

  for (const filePath of candidatePaths) {
    try {
      await fs.access(filePath);
      return filePath;
    } catch {}
  }

  return writablePath;
}

async function getWritableStorePath() {
  for (const dirPath of STORE_DIR_CANDIDATES) {
    try {
      await fs.mkdir(dirPath, { recursive: true });
      await fs.access(dirPath, fs.constants.W_OK);
      return path.join(dirPath, STORE_FILE_NAME);
    } catch {}
  }

  return path.join('/tmp', STORE_FILE_NAME);
}

export async function readComments() {
  if (canUseKv()) {
    return readCommentsFromKv();
  }

  const filePath = await getReadableStorePath();

  try {
    const raw = await fs.readFile(filePath, 'utf8');
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];

    return parsed
      .map(normalizeComment)
      .filter(Boolean)
      .sort((left, right) => left.ts - right.ts);
  } catch {
    return [];
  }
}

export async function writeComments(comments) {
  if (canUseKv()) {
    return writeCommentsToKv(comments);
  }

  const filePath = await getWritableStorePath();
  const normalized = Array.isArray(comments)
    ? comments.map(normalizeComment).filter(Boolean).sort((left, right) => left.ts - right.ts)
    : [];

  await fs.writeFile(filePath, JSON.stringify(normalized, null, 2), 'utf8');
  return normalized;
}

export async function appendComment(comment) {
  if (canUseKv()) {
    return appendCommentToKv(comment);
  }

  const normalized = normalizeComment(comment);
  if (!normalized) {
    throw new Error('Invalid comment payload');
  }

  const comments = await readComments();
  if (comments.some(entry => entry.id === normalized.id)) {
    return comments;
  }

  comments.push(normalized);
  return writeComments(comments);
}

export async function clearComments() {
  if (canUseKv()) {
    return clearCommentsFromKv();
  }

  return writeComments([]);
}