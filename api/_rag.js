import { promises as fs } from 'fs';
import path from 'path';

const OPENAI_BASE_URL = 'https://api.openai.com/v1';
const CHAT_MODEL = 'gpt-4o-mini';
const EMBEDDING_MODEL = 'text-embedding-3-small';
const DB_KEY = '__greenroom_rag_db__';
const WORKSPACE_ROOT = process.cwd();
const WORKSPACE_DOC_DIRS = ['.rag-docs', 'writings', 'knowledge', 'docs'];
const TEXT_FILE_EXTENSIONS = new Set(['.txt', '.md', '.markdown', '.text']);
const LOCAL_EMBEDDING_DIMS = 256;

function getSearchRoots() {
  return [WORKSPACE_ROOT];
}

function getDb() {
  if (!globalThis[DB_KEY]) {
    globalThis[DB_KEY] = {
      documents: [],
      chunks: [],
      conversations: {},
      workspaceSignature: null
    };
  }
  return globalThis[DB_KEY];
}

function requireApiKey() {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is not set');
  }
  return apiKey;
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const len = Math.min(a.length, b.length);

  for (let i = 0; i < len; i++) {
    const av = a[i];
    const bv = b[i];
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }

  if (!normA || !normB) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function tokenize(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .map(token => token.trim())
    .filter(token => token.length > 2);
}

function lexicalScore(query, text) {
  const queryTokens = tokenize(query);
  const textTokensArray = tokenize(text);
  const textTokens = new Set(textTokensArray);

  if (!queryTokens.length || !textTokens.size) return 0;

  let matches = 0;
  for (const token of queryTokens) {
    if (textTokens.has(token)) {
      matches += 1;
      continue;
    }

    if (token.length >= 4) {
      const hasPrefixMatch = textTokensArray.some(textToken => {
        if (textToken.length < 4) return false;
        return textToken.startsWith(token) || token.startsWith(textToken);
      });
      if (hasPrefixMatch) matches += 1;
    }
  }

  return matches / queryTokens.length;
}

function hashToken(token) {
  let hash = 2166136261;
  for (let i = 0; i < token.length; i++) {
    hash ^= token.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash >>> 0);
}

function buildLocalEmbedding(text) {
  const vector = new Array(LOCAL_EMBEDDING_DIMS).fill(0);
  const tokens = tokenize(text);

  if (!tokens.length) return vector;

  for (const token of tokens) {
    const hash = hashToken(token);
    const index = hash % LOCAL_EMBEDDING_DIMS;
    vector[index] += 1;
  }

  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (!norm) return vector;

  return vector.map(value => value / norm);
}

function splitIntoSentences(text) {
  return String(text || '')
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(Boolean);
}

function chunkText(rawText, maxChars = 900, overlapChars = 180) {
  const normalized = String(rawText || '').replace(/\r\n/g, '\n').trim();
  if (!normalized) return [];

  const paragraphs = normalized
    .split(/\n{2,}/)
    .map(p => p.replace(/\s+/g, ' ').trim())
    .filter(Boolean);

  const chunks = [];
  for (const p of paragraphs) {
    if (p.length <= maxChars) {
      chunks.push(p);
      continue;
    }

    const sentences = splitIntoSentences(p);
    let current = '';

    for (const sentence of sentences) {
      const proposed = current ? `${current} ${sentence}` : sentence;
      if (proposed.length <= maxChars) {
        current = proposed;
      } else {
        if (current) chunks.push(current);
        const tail = current.slice(Math.max(0, current.length - overlapChars)).trim();
        current = tail ? `${tail} ${sentence}` : sentence;
      }
    }

    if (current) chunks.push(current);
  }

  return chunks.filter(Boolean);
}

async function openAiJson(path, payload) {
  const apiKey = requireApiKey();
  const response = await fetch(`${OPENAI_BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify(payload)
  });

  const data = await response.json();
  if (!response.ok) {
    const message = data?.error?.message || 'OpenAI request failed';
    throw new Error(message);
  }
  return data;
}

async function embedTexts(texts) {
  if (!canUseOpenAi()) {
    return texts.map(text => buildLocalEmbedding(text));
  }

  const data = await openAiJson('/embeddings', {
    model: EMBEDDING_MODEL,
    input: texts
  });
  return data.data.map(d => d.embedding);
}

function canUseOpenAi() {
  return Boolean(process.env.OPENAI_API_KEY);
}

function selectBestSentences(message, contextChunks) {
  const candidates = [];

  contextChunks.forEach(chunk => {
    splitIntoSentences(chunk.text).forEach(sentence => {
      const score = lexicalScore(message, sentence);
      if (score > 0) {
        candidates.push({ sentence, score });
      }
    });
  });

  const unique = [];
  const seen = new Set();
  for (const item of candidates.sort((a, b) => b.score - a.score)) {
    const key = item.sentence.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(item.sentence.trim());
    if (unique.length >= 2) break;
  }

  return unique;
}

function selectBestExcerpt(message, contextChunks) {
  const candidates = [];

  contextChunks.forEach(chunk => {
    const rawParts = String(chunk.text || '')
      .split(/\n+/)
      .map(part => part.trim())
      .filter(Boolean);

    rawParts.forEach(part => {
      const score = lexicalScore(message, part);
      if (score > 0 && part.length >= 30) {
        candidates.push({ text: part, score });
      }
    });
  });

  if (!candidates.length) return '';

  candidates.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return b.text.length - a.text.length;
  });

  return candidates[0].text;
}

function buildLocalAnswer(message, contextChunks) {
  if (!contextChunks.length) {
    return 'I do not have enough uploaded material yet.';
  }

  const bestSentences = selectBestSentences(message, contextChunks);
  if (bestSentences.length) {
    return limitReply(bestSentences.join(' '));
  }

  const excerpt = selectBestExcerpt(message, contextChunks);
  if (excerpt) {
    return limitReply(excerpt);
  }

  const fallbackLines = String(contextChunks[0]?.text || '')
    .split(/\n+/)
    .map(line => line.trim())
    .filter(line => {
      if (!line) return false;
      if (line.length < 20) return false;
      if (/^section\s+\d+/i.test(line)) return false;
      if (/^[a-z]+:\s*$/i.test(line)) return false;
      return true;
    })
    .slice(0, 2);

  if (fallbackLines.length) {
    return limitReply(fallbackLines.join(' '));
  }

  return 'I do not have enough uploaded material yet.';
}

function getConversation(sessionId) {
  const db = getDb();
  if (!db.conversations[sessionId]) {
    db.conversations[sessionId] = {
      lastContextChunkIds: [],
      lastUserQuestion: ''
    };
  }
  return db.conversations[sessionId];
}

function quickFollowUpHeuristic(question) {
  const q = String(question || '').trim().toLowerCase();
  if (!q) return 'retrieve-new';

  const followUpCues = [
    'what about', 'and what', 'and how', 'can you expand', 'can you elaborate',
    'tell me more', 'go deeper', 'that part', 'this part', 'those', 'it', 'they',
    'he', 'she', 'them', 'more on that', 'why is that'
  ];

  if (q.length < 90 && followUpCues.some(cue => q.includes(cue))) {
    return 'reuse-context';
  }

  const newQueryCues = ['summarize', 'define', 'what is', 'who is', 'when did', 'where is'];
  if (newQueryCues.some(cue => q.includes(cue))) {
    return 'retrieve-new';
  }

  return 'retrieve-new';
}

async function decideQueryMode({ question, history, hasPreviousContext }) {
  if (!hasPreviousContext) return 'retrieve-new';

  const heuristic = quickFollowUpHeuristic(question);
  const shortHistory = (history || []).slice(-6);

  try {
    const classifierMessages = [
      {
        role: 'system',
        content: 'Decide if the new user question should reuse previous retrieval context or run a new retrieval. Reply with exactly one token: reuse-context or retrieve-new.'
      },
      {
        role: 'user',
        content: `Recent chat:\n${JSON.stringify(shortHistory)}\n\nNew question:\n${question}\n\nHeuristic suggestion: ${heuristic}`
      }
    ];

    const data = await openAiJson('/chat/completions', {
      model: CHAT_MODEL,
      messages: classifierMessages,
      max_tokens: 8,
      temperature: 0
    });

    const label = (data?.choices?.[0]?.message?.content || '').trim().toLowerCase();
    if (label.includes('reuse-context')) return 'reuse-context';
    return 'retrieve-new';
  } catch {
    return heuristic;
  }
}

function limitReply(text) {
  const compact = String(text || '').replace(/\r\n/g, '\n').trim();
  const sentenceParts = compact
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(Boolean)
    .slice(0, 2);

  let out = sentenceParts.join(' ');
  if (!out) out = compact;

  const lines = out
    .split('\n')
    .map(l => l.trim())
    .filter(Boolean)
    .slice(0, 3);

  return lines.join('\n');
}

function buildContext(chunks) {
  if (!chunks.length) return 'No context available.';

  return chunks
    .map((chunk, idx) => {
      return `[${idx + 1}] Title: ${chunk.title}\n${chunk.text}`;
    })
    .join('\n\n');
}

function contextMatchesQuestion(question, chunks, minScore = 0.2) {
  if (!chunks.length) return false;
  const merged = chunks.map(chunk => `${chunk.title} ${chunk.text}`).join(' ');
  return lexicalScore(question, merged) >= minScore;
}

function filterChunksByMentionedTitle(question, chunks) {
  if (!chunks.length) return chunks;

  const queryTokens = new Set(tokenize(question));
  if (!queryTokens.size) return chunks;

  const matchedTitles = new Set(
    chunks
      .map(chunk => chunk.title)
      .filter(title => {
        const titleBase = String(title || '').replace(/\.[^.]+$/, '');
        const titleTokens = tokenize(titleBase);
        return titleTokens.some(token => queryTokens.has(token));
      })
  );

  if (!matchedTitles.size) return chunks;
  return chunks.filter(chunk => matchedTitles.has(chunk.title));
}

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function walkDirectory(dirPath) {
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const entryPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      files.push(...await walkDirectory(entryPath));
      continue;
    }

    if (TEXT_FILE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      files.push(entryPath);
    }
  }

  return files;
}

async function readWorkspaceDocumentFiles() {
  const sources = [];
  const seenPaths = new Set();
  const searchRoots = getSearchRoots();

  for (const root of searchRoots) {
    for (const dirName of WORKSPACE_DOC_DIRS) {
      const absDir = path.join(root, dirName);
      if (!await pathExists(absDir)) continue;

      const files = await walkDirectory(absDir);
      for (const filePath of files) {
        const canonicalPath = path.resolve(filePath);
        if (seenPaths.has(canonicalPath)) continue;
        seenPaths.add(canonicalPath);

        const content = await fs.readFile(filePath, 'utf8');
        const relativePath = path.relative(root, filePath);
        if (!String(content).trim()) continue;

        sources.push({
          title: path.basename(filePath),
          content,
          sourcePath: relativePath
        });
      }
    }
  }

  return sources;
}

async function listWorkspaceSourceFiles() {
  const sources = [];
  const seenPaths = new Set();
  const searchRoots = getSearchRoots();

  for (const root of searchRoots) {
    for (const dirName of WORKSPACE_DOC_DIRS) {
      const absDir = path.join(root, dirName);
      if (!await pathExists(absDir)) continue;

      const files = await walkDirectory(absDir);
      for (const filePath of files) {
        const canonicalPath = path.resolve(filePath);
        if (seenPaths.has(canonicalPath)) continue;
        seenPaths.add(canonicalPath);

        const stats = await fs.stat(filePath);
        const relativePath = path.relative(root, filePath);
        sources.push({
          title: path.basename(filePath),
          sourcePath: relativePath,
          size: stats.size
        });
      }
    }
  }

  return sources;
}

async function getWorkspaceSignature() {
  const parts = [];
  const seenPaths = new Set();
  const searchRoots = getSearchRoots();

  for (const root of searchRoots) {
    for (const dirName of WORKSPACE_DOC_DIRS) {
      const absDir = path.join(root, dirName);
      if (!await pathExists(absDir)) continue;

      const files = await walkDirectory(absDir);
      for (const filePath of files) {
        const canonicalPath = path.resolve(filePath);
        if (seenPaths.has(canonicalPath)) continue;
        seenPaths.add(canonicalPath);

        const stats = await fs.stat(filePath);
        const relativePath = path.relative(root, filePath);
        parts.push(`${relativePath}:${stats.size}:${stats.mtimeMs}`);
      }
    }
  }

  return parts.sort().join('|');
}

async function rebuildDatabaseFromSources(sources) {
  const db = getDb();
  db.documents = [];
  db.chunks = [];
  db.conversations = {};

  const results = [];
  for (const source of sources) {
    const result = await uploadDocument({
      title: source.title,
      content: source.content,
      sourcePath: source.sourcePath
    });
    results.push(result);
  }

  return results;
}

export async function uploadDocument({ title, content, sourcePath = null }) {
  const cleanTitle = String(title || '').trim() || `doc-${Date.now()}`;
  const cleanContent = String(content || '').trim();

  if (!cleanContent) {
    throw new Error('No document content provided');
  }

  const chunks = chunkText(cleanContent);
  if (!chunks.length) {
    throw new Error('Could not create chunks from document content');
  }

  let embeddings;
  let embeddingProvider = 'local';

  try {
    embeddings = await embedTexts(chunks);
    embeddingProvider = canUseOpenAi() ? 'openai' : 'local';
  } catch {
    embeddings = chunks.map(text => buildLocalEmbedding(text));
    embeddingProvider = 'local';
  }
  const db = getDb();
  const documentId = `doc_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  const chunkRecords = chunks.map((text, index) => ({
    id: `${documentId}_chunk_${index + 1}`,
    documentId,
    title: cleanTitle,
    text,
    embedding: embeddings[index],
    createdAt: Date.now()
  }));

  db.documents.push({
    id: documentId,
    title: cleanTitle,
    sourcePath,
    createdAt: Date.now(),
    chunkCount: chunkRecords.length,
    embeddingBacked: true,
    embeddingProvider
  });
  db.chunks.push(...chunkRecords);

  return {
    documentId,
    title: cleanTitle,
    sourcePath,
    chunkCount: chunkRecords.length,
    embeddingBacked: true,
    embeddingProvider
  };
}

export async function syncWorkspaceDocuments() {
  const signature = await getWorkspaceSignature();
  if (!signature) {
    throw new Error('No workspace documents found. Add .txt or .md files under .rag-docs/, writings/, knowledge/, or docs/.');
  }

  const sources = await readWorkspaceDocumentFiles();
  const results = await rebuildDatabaseFromSources(sources);
  const db = getDb();
  db.workspaceSignature = signature;

  return {
    indexedCount: results.length,
    documents: results.map(doc => ({
      title: doc.title,
      sourcePath: doc.sourcePath,
      chunkCount: doc.chunkCount,
      embeddingBacked: doc.embeddingBacked,
      embeddingProvider: doc.embeddingProvider
    }))
  };
}

export async function inspectWorkspaceState() {
  const sourceFiles = await listWorkspaceSourceFiles();
  return {
    apiKeyPresent: Boolean(process.env.OPENAI_API_KEY),
    searchRoots: getSearchRoots(),
    sourceFiles
  };
}

export async function ensureWorkspaceDocumentsIndexed() {
  const db = getDb();

  const signature = await getWorkspaceSignature();

  if (!signature) {
    db.documents = [];
    db.chunks = [];
    db.conversations = {};
    db.workspaceSignature = null;
    return [];
  }

  if (db.workspaceSignature === signature && db.documents.length > 0 && db.chunks.length > 0) {
    return db.documents;
  }

  await syncWorkspaceDocuments();
  return db.documents;
}

export async function listDocuments() {
  await ensureWorkspaceDocumentsIndexed();
  const db = getDb();
  return db.documents;
}

export async function answerWithRag({ message, history = [], sessionId = 'default' }) {
  await ensureWorkspaceDocumentsIndexed();

  const db = getDb();
  const conversation = getConversation(sessionId);
  const hasPreviousContext = conversation.lastContextChunkIds.length > 0;

  const mode = await decideQueryMode({
    question: message,
    history,
    hasPreviousContext
  });

  let contextChunks = [];

  if (mode === 'reuse-context') {
    contextChunks = db.chunks.filter(c => conversation.lastContextChunkIds.includes(c.id));
    if (!contextMatchesQuestion(message, contextChunks)) {
      contextChunks = [];
      conversation.lastContextChunkIds = [];
    }
  }

  if (!contextChunks.length) {
    const vectorChunks = db.chunks.filter(chunk => Array.isArray(chunk.embedding));
    if (vectorChunks.length > 0) {
      try {
        const [questionEmbedding] = await embedTexts([message]);
        const scored = vectorChunks
          .map(chunk => ({
            ...chunk,
            score: cosineSimilarity(questionEmbedding, chunk.embedding)
          }))
          .sort((a, b) => b.score - a.score)
          .slice(0, 6)
          .filter(c => c.score > 0.15);

        contextChunks = scored;
        conversation.lastContextChunkIds = scored.map(c => c.id);
      } catch {
        // If vector query fails (e.g. bad API key), lexical retrieval below still runs.
      }
    }
  }

  if (!contextChunks.length && db.chunks.length > 0) {
    const lexicalMatches = db.chunks
      .map(chunk => ({
        ...chunk,
        score: lexicalScore(message, `${chunk.title} ${chunk.text}`)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 6);

    contextChunks = lexicalMatches;
    conversation.lastContextChunkIds = lexicalMatches.map(chunk => chunk.id);
  }

  contextChunks = filterChunksByMentionedTitle(message, contextChunks);
  if (contextChunks.length > 0) {
    conversation.lastContextChunkIds = contextChunks.map(chunk => chunk.id);
  }

  conversation.lastUserQuestion = message;

  const context = buildContext(contextChunks);
  const promptMessages = [
    {
      role: 'system',
      content: 'You are greenroom, a retrieval-augmented assistant. Use the provided context first. If the answer is not in context, say you do not have enough uploaded material yet. Reply in at most 2 sentences and no more than 3 short lines.'
    },
    {
      role: 'system',
      content: `Retrieved context:\n${context}`
    },
    ...history.slice(-6),
    { role: 'user', content: message }
  ];

  let reply = buildLocalAnswer(message, contextChunks);

  if (canUseOpenAi()) {
    try {
      const completion = await openAiJson('/chat/completions', {
        model: CHAT_MODEL,
        messages: promptMessages,
        max_tokens: 180,
        temperature: 0.3
      });
      reply = limitReply(completion?.choices?.[0]?.message?.content || reply);
    } catch {
      // Keep local fallback reply when OpenAI generation fails.
    }
  }

  const citations = contextChunks.map(c => ({
    documentId: c.documentId,
    title: c.title,
    score: Number(c.score?.toFixed ? c.score.toFixed(4) : 0)
  }));

  return {
    reply,
    mode,
    citations
  };
}
