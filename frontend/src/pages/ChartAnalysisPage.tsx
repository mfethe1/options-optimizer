import React, { useState, useRef } from 'react';
import { analyzeChart, AnalysisType, ChartAnalysisResponse } from '../services/visionApi';
import toast from 'react-hot-toast';

const ChartAnalysisPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [analysisType, setAnalysisType] = useState<AnalysisType>('comprehensive');
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ChartAnalysisResponse | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      toast.error('Please select an image file');
      return;
    }

    setSelectedFile(file);
    setResult(null);

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      toast.error('Please select a chart image');
      return;
    }

    setLoading(true);
    try {
      const response = await analyzeChart(
        selectedFile,
        analysisType,
        question || undefined
      );
      setResult(response);
      toast.success('Chart analyzed successfully!');
    } catch (error: any) {
      toast.error(error.message || 'Failed to analyze chart');
      console.error('Error analyzing chart:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity?: string) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return 'text-red-600';
      case 'medium':
        return 'text-yellow-600';
      case 'low':
        return 'text-green-600';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">
          AI Chart Analysis
        </h1>
        <p className="text-gray-600 mt-2">
          Upload chart screenshots for AI-powered pattern recognition and analysis
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="space-y-4">
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Upload Chart</h2>

            {/* Drag and Drop Zone */}
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
            >
              {preview ? (
                <div className="space-y-4">
                  <img
                    src={preview}
                    alt="Chart preview"
                    className="max-h-64 mx-auto rounded"
                  />
                  <p className="text-sm text-gray-600">{selectedFile?.name}</p>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedFile(null);
                      setPreview(null);
                      setResult(null);
                    }}
                    className="text-red-600 hover:text-red-700 text-sm"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <div>
                  <svg
                    className="w-12 h-12 mx-auto text-gray-400 mb-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                  <p className="text-gray-600 mb-2">
                    Drag and drop a chart image here, or click to select
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports PNG, JPG, WEBP
                  </p>
                </div>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInputChange}
              className="hidden"
            />
          </div>

          {/* Analysis Options */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Analysis Settings</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Analysis Type
                </label>
                <select
                  value={analysisType}
                  onChange={(e) => setAnalysisType(e.target.value as AnalysisType)}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="comprehensive">Comprehensive (Full Analysis)</option>
                  <option value="pattern">Pattern Recognition</option>
                  <option value="levels">Support/Resistance Levels</option>
                  <option value="flow">Options Flow Analysis</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Specific Question (Optional)
                </label>
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="e.g., Is this bullish or bearish?"
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <button
                onClick={handleAnalyze}
                disabled={!selectedFile || loading}
                className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-semibold"
              >
                {loading ? 'Analyzing...' : 'Analyze Chart'}
              </button>
            </div>
          </div>

          {/* Info */}
          <div className="bg-blue-50 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">
              💡 How it works
            </h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Upload charts from TradingView, ThinkOrSwim, or anywhere</li>
              <li>• AI identifies patterns, levels, and indicators</li>
              <li>• Get actionable trading recommendations</li>
              <li>• Powered by GPT-4 Vision & Claude 3.5 Sonnet</li>
            </ul>
          </div>
        </div>

        {/* Results Section */}
        <div className="space-y-4">
          {result ? (
            <>
              {/* Analysis Summary */}
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">Analysis Results</h2>
                  <span className="text-xs text-gray-500">
                    Provider: {result.provider}
                  </span>
                </div>

                {/* Patterns */}
                {result.analysis.patterns && result.analysis.patterns.length > 0 && (
                  <div className="mb-6">
                    <h3 className="font-semibold text-gray-900 mb-3">Patterns</h3>
                    <div className="space-y-2">
                      {result.analysis.patterns.map((pattern, idx) => (
                        <div key={idx} className="border border-gray-200 rounded-lg p-3">
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-medium">{pattern.type}</span>
                            <span className={`text-sm ${pattern.bias === 'bullish' ? 'text-green-600' : 'text-red-600'}`}>
                              {pattern.bias}
                            </span>
                          </div>
                          <div className="text-sm text-gray-600">
                            Confidence: {Math.round(pattern.confidence * 100)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Levels */}
                {result.analysis.levels && (
                  <div className="mb-6">
                    <h3 className="font-semibold text-gray-900 mb-3">Key Levels</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-sm text-gray-600 mb-2">Support</div>
                        <div className="space-y-1">
                          {result.analysis.levels.support?.map((level, idx) => (
                            <div key={idx} className="text-green-600 font-mono">
                              ${level.toFixed(2)}
                            </div>
                          ))}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-600 mb-2">Resistance</div>
                        <div className="space-y-1">
                          {result.analysis.levels.resistance?.map((level, idx) => (
                            <div key={idx} className="text-red-600 font-mono">
                              ${level.toFixed(2)}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Trend */}
                {result.analysis.trend && (
                  <div className="mb-6">
                    <h3 className="font-semibold text-gray-900 mb-3">Trend</h3>
                    <div className="flex items-center gap-4">
                      <div className="flex-1">
                        <div className="text-sm text-gray-600">Direction</div>
                        <div className="font-medium">{result.analysis.trend.direction}</div>
                      </div>
                      <div className="flex-1">
                        <div className="text-sm text-gray-600">Strength</div>
                        <div className="font-medium">{result.analysis.trend.strength}</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Indicators */}
                {result.analysis.indicators && Object.keys(result.analysis.indicators).length > 0 && (
                  <div className="mb-6">
                    <h3 className="font-semibold text-gray-900 mb-3">Indicators</h3>
                    <div className="space-y-2">
                      {Object.entries(result.analysis.indicators).map(([key, value]) => (
                        <div key={key} className="flex justify-between text-sm">
                          <span className="text-gray-600 uppercase">{key}</span>
                          <span className="font-medium">{value as string}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recommendation */}
                {result.analysis.recommendation && (
                  <div className="bg-blue-50 rounded-lg p-4">
                    <h3 className="font-semibold text-blue-900 mb-2">
                      Recommendation
                    </h3>
                    <div className="text-sm text-blue-800 space-y-1">
                      <div>
                        <strong>Action:</strong> {result.analysis.recommendation.action}
                      </div>
                      {result.analysis.recommendation.strikes && (
                        <div>
                          <strong>Strikes:</strong>{' '}
                          {result.analysis.recommendation.strikes.join(', ')}
                        </div>
                      )}
                      {result.analysis.recommendation.expiration && (
                        <div>
                          <strong>Expiration:</strong>{' '}
                          {result.analysis.recommendation.expiration}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Risks */}
                {result.analysis.risks && result.analysis.risks.length > 0 && (
                  <div className="mt-4 bg-yellow-50 rounded-lg p-4">
                    <h3 className="font-semibold text-yellow-900 mb-2">
                      ⚠️ Risks
                    </h3>
                    <ul className="text-sm text-yellow-800 space-y-1">
                      {result.analysis.risks.map((risk, idx) => (
                        <li key={idx}>• {risk}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="bg-white rounded-lg shadow p-12 text-center">
              <svg
                className="w-16 h-16 mx-auto text-gray-400 mb-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
              <p className="text-gray-600">
                Upload and analyze a chart to see results here
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChartAnalysisPage;
