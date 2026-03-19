import { promises as fs } from 'fs';
import path from 'path';

const STORE_DIR_CANDIDATES = [
  path.join(process.cwd(), '.data'),
  '/tmp'
];

const STORE_FILE_NAME = 'greenroom-comments.json';

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
  const filePath = await getWritableStorePath();
  const normalized = Array.isArray(comments)
    ? comments.map(normalizeComment).filter(Boolean).sort((left, right) => left.ts - right.ts)
    : [];

  await fs.writeFile(filePath, JSON.stringify(normalized, null, 2), 'utf8');
  return normalized;
}

export async function appendComment(comment) {
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
  return writeComments([]);
}