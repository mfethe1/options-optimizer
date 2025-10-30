import React, { useState, useRef, useEffect } from 'react';
import {
  sendConversationMessage,
  ConversationResponse,
  getSupportedIntents,
  getExplanation
} from '../services/conversationalApi';
import toast from 'react-hot-toast';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  intent?: string;
  confidence?: number;
  data?: any;
}

const ConversationalTradingPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const [intents, setIntents] = useState<any>(null);
  const [showExamples, setShowExamples] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const userId = 'demo_user'; // In production, get from auth

  useEffect(() => {
    // Generate session ID
    setSessionId(`session_${Date.now()}`);

    // Load supported intents
    getSupportedIntents()
      .then(data => setIntents(data))
      .catch(err => console.error('Failed to load intents:', err));
  }, []);

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setShowExamples(false);

    try {
      const response: ConversationResponse = await sendConversationMessage({
        message: input,
        user_id: userId,
        session_id: sessionId,
      });

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        intent: response.intent,
        confidence: response.confidence,
        data: response.data,
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Update session ID if it changed
      if (response.session_id !== sessionId) {
        setSessionId(response.session_id);
      }
    } catch (error: any) {
      toast.error(error.message || 'Failed to send message');
      console.error('Error sending message:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleExampleClick = (example: string) => {
    setInput(example);
    setShowExamples(false);
  };

  const getIntentColor = (intent: string) => {
    const colors: Record<string, string> = {
      trade_execution: 'bg-red-100 text-red-800',
      risk_analysis: 'bg-yellow-100 text-yellow-800',
      research: 'bg-blue-100 text-blue-800',
      portfolio_review: 'bg-purple-100 text-purple-800',
      education: 'bg-green-100 text-green-800',
      market_data: 'bg-indigo-100 text-indigo-800',
      general: 'bg-gray-100 text-gray-800',
    };
    return colors[intent] || 'bg-gray-100 text-gray-800';
  };

  const exampleQueries = [
    "What's the risk/reward on selling NVDA 950 puts expiring next Friday?",
    "Show me high IV stocks in the tech sector",
    "What happens if AAPL drops 10% before my calls expire?",
    "Explain iron condors",
    "What's TSLA trading at?",
    "Find stocks with earnings this week",
  ];

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">
          Conversational Trading
        </h1>
        <p className="text-gray-600 mt-2">
          Ask questions in natural language. I'll help with trades, analysis, research, and education.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chat Interface */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-lg h-[600px] flex flex-col">
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.length === 0 && showExamples && (
                <div className="text-center py-12">
                  <div className="text-gray-400 mb-4">
                    <svg
                      className="w-16 h-16 mx-auto mb-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                      />
                    </svg>
                  </div>
                  <h3 className="text-lg font-semibold text-gray-700 mb-2">
                    Start a conversation
                  </h3>
                  <p className="text-gray-500 mb-6">
                    Try one of these examples:
                  </p>
                  <div className="space-y-2">
                    {exampleQueries.map((query, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleExampleClick(query)}
                        className="block w-full text-left px-4 py-3 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors text-sm text-blue-700"
                      >
                        {query}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((message, idx) => (
                <div
                  key={idx}
                  className={`flex ${
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg px-4 py-3 ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-900'
                    }`}
                  >
                    {message.role === 'assistant' && message.intent && (
                      <div className="mb-2 flex items-center gap-2">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getIntentColor(message.intent)}`}>
                          {message.intent.replace('_', ' ')}
                        </span>
                        <span className="text-xs text-gray-500">
                          {Math.round((message.confidence || 0) * 100)}% confident
                        </span>
                      </div>
                    )}
                    <div className="whitespace-pre-wrap">{message.content}</div>
                    {message.data && (
                      <div className="mt-3 pt-3 border-t border-gray-300">
                        <div className="text-xs font-semibold text-gray-600 mb-2">
                          Data:
                        </div>
                        <pre className="text-xs bg-white rounded p-2 overflow-x-auto">
                          {JSON.stringify(message.data, null, 2)}
                        </pre>
                      </div>
                    )}
                    <div className="text-xs opacity-70 mt-1">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}

              {loading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 rounded-lg px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-gray-200 p-4">
              <div className="flex gap-2">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me anything about options trading..."
                  className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                  rows={2}
                  disabled={loading}
                />
                <button
                  onClick={handleSend}
                  disabled={loading || !input.trim()}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Intent Guide */}
        <div className="space-y-4">
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="font-semibold text-gray-900 mb-3">
              What I can help with
            </h3>
            <div className="space-y-2 text-sm">
              {intents?.intents && Object.entries(intents.intents).map(([key, value]: [string, any]) => (
                <div key={key} className="border-l-4 border-blue-500 pl-3 py-1">
                  <div className="font-medium text-gray-900">
                    {value.description}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    e.g., "{value.examples[0]}"
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-blue-50 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">
              💡 Tips
            </h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Be specific with symbols and dates</li>
              <li>• Ask follow-up questions</li>
              <li>• Request explanations anytime</li>
              <li>• I remember context in the conversation</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConversationalTradingPage;
