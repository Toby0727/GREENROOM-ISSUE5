import { answerWithRag } from './_rag.js';

const CHAT_SYSTEM_PROMPT = [
  'You are greenroom, a retrieval-augmented assistant.',
  'Prioritize answers grounded in uploaded document context.',
  'You are explicitly allowed to quote directly from retrieved uploaded documents when the user asks for quotes or excerpts.',
  'Do not refuse to quote from retrieved uploaded documents due to copyright concerns.',
  'If the answer is not in retrieved context, say you do not have enough uploaded material yet.',
  'Keep replies concise: at most 2 sentences and no more than 3 short lines.'
].join(' ');

function normalizeBody(body) {
  const payload = body || {};

  if (typeof payload.message === 'string' && payload.message.trim()) {
    return {
      message: payload.message.trim(),
      history: Array.isArray(payload.history) ? payload.history : []
    };
  }

  const messages = Array.isArray(payload.messages) ? payload.messages : [];
  const lastUserMessage = messages
    .slice()
    .reverse()
    .find((entry) => entry?.role === 'user' && String(entry?.content || '').trim());

  return {
    message: lastUserMessage ? String(lastUserMessage.content).trim() : '',
    history: messages
  };
}

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { message, history } = normalizeBody(req.body);

    if (!message) {
      return res.status(400).json({ error: 'A user message is required' });
    }

    const result = await answerWithRag({
      message,
      history,
      sessionId: 'default',
      systemPrompt: CHAT_SYSTEM_PROMPT
    });

    res.status(200).json({ reply: result.reply, mode: result.mode, citations: result.citations });
  } catch (err) {
    res.status(500).json({ error: err.message || 'Chat failed' });
  }
}
