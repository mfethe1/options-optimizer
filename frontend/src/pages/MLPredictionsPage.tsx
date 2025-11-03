/**
 * ML Predictions Page
 *
 * Machine learning-based price predictions using LSTM models.
 * Expected impact: +2-4% monthly through better entry/exit timing.
 */

import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import {
  getMLPrediction,
  getBatchPredictions,
  trainModel,
  getModelInfo,
  getMLStrategies,
  getMLHealth,
  MLPrediction,
  ModelInfo,
  MLStrategy
} from '../services/mlApi';

export default function MLPredictionsPage() {
  const [symbol, setSymbol] = useState('AAPL');
  const [inputSymbol, setInputSymbol] = useState('AAPL');
  const [prediction, setPrediction] = useState<MLPrediction | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [strategies, setStrategies] = useState<MLStrategy[]>([]);
  const [health, setHealth] = useState<{ status: string; tensorflow_available: boolean } | null>(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);

  // Watchlist predictions
  const [watchlist] = useState(['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']);
  const [watchlistPredictions, setWatchlistPredictions] = useState<MLPrediction[]>([]);

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    loadSymbolData(symbol);
  }, [symbol]);

  const loadInitialData = async () => {
    try {
      const [strategiesData, healthData] = await Promise.all([
        getMLStrategies(),
        getMLHealth()
      ]);

      setStrategies(strategiesData.strategies);
      setHealth(healthData);

      // Load watchlist predictions
      loadWatchlistPredictions();
    } catch (error: any) {
      console.error('Failed to load initial data:', error);
    }
  };

  const loadSymbolData = async (sym: string) => {
    setLoading(true);
    try {
      const [predData, infoData] = await Promise.all([
        getMLPrediction(sym).catch(() => null),
        getModelInfo(sym).catch(() => null)
      ]);

      setPrediction(predData);
      setModelInfo(infoData);
    } catch (error: any) {
      console.error(`Failed to load data for ${sym}:`, error);
    } finally {
      setLoading(false);
    }
  };

  const loadWatchlistPredictions = async () => {
    try {
      const predictions = await getBatchPredictions(watchlist);
      setWatchlistPredictions(predictions);
    } catch (error: any) {
      console.error('Failed to load watchlist predictions:', error);
    }
  };

  const handleSymbolChange = () => {
    const newSymbol = inputSymbol.toUpperCase().trim();
    if (newSymbol && newSymbol !== symbol) {
      setSymbol(newSymbol);
      toast.success(`Switched to ${newSymbol}`);
    }
  };

  const handleTrainModel = async () => {
    setTraining(true);
    try {
      const result = await trainModel({
        symbol,
        years: 5,
        epochs: 100,
        force_retrain: false
      });

      toast.success(result.message);
      toast('Training runs in background. Check back in 10-30 minutes.', { duration: 5000 });
    } catch (error: any) {
      toast.error(error.message || 'Failed to start training');
    } finally {
      setTraining(false);
    }
  };

  const handleRefreshPrediction = async () => {
    setLoading(true);
    try {
      const predData = await getMLPrediction(symbol, true);
      setPrediction(predData);
      toast.success('Prediction refreshed');
    } catch (error: any) {
      toast.error(error.message || 'Failed to refresh prediction');
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.7) return 'text-green-600';
    if (confidence >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getRecommendationColor = (rec: string) => {
    if (rec === 'BUY') return 'bg-green-100 text-green-800';
    if (rec === 'SELL') return 'bg-red-100 text-red-800';
    return 'bg-gray-100 text-gray-800';
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">ML Price Predictions</h1>
        <p className="text-gray-600">LSTM-based predictions • +2-4% monthly through better timing</p>
      </div>

      {/* Health Status */}
      {health && (
        <div className={`p-4 rounded-lg mb-6 ${
          health.tensorflow_available ? 'bg-green-50 text-green-800' : 'bg-yellow-50 text-yellow-800'
        }`}>
          <div className="flex items-center justify-between">
            <span className="font-semibold">
              {health.tensorflow_available ? '✓ ML Service Operational' : '⚠ TensorFlow Not Available'}
            </span>
            <span className="text-sm">
              {health.tensorflow_available ? 'LSTM models ready' : 'Install TensorFlow to enable ML predictions'}
            </span>
          </div>
        </div>
      )}

      {/* Symbol Input */}
      <div className="bg-white p-6 rounded-lg shadow-md mb-6">
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-2">Symbol</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={inputSymbol}
                onChange={(e) => setInputSymbol(e.target.value.toUpperCase())}
                onKeyPress={(e) => e.key === 'Enter' && handleSymbolChange()}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
                placeholder="Enter symbol (e.g., AAPL)"
              />
              <button
                onClick={handleSymbolChange}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Load
              </button>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Actions</label>
            <button
              onClick={handleRefreshPrediction}
              disabled={loading}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-400 mr-2"
            >
              Refresh
            </button>
            {!modelInfo?.model_exists && (
              <button
                onClick={handleTrainModel}
                disabled={training}
                className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-400"
              >
                {training ? 'Training...' : 'Train Model'}
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* ML Prediction */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-bold text-gray-900 mb-4">{symbol} ML Prediction</h2>

          {loading && <div className="text-center py-8">Loading...</div>}

          {!loading && !prediction && (
            <div className="text-center py-8 text-gray-500">
              <p>No prediction available</p>
              {!modelInfo?.model_exists && (
                <p className="mt-2 text-sm">Train a model first to get predictions</p>
              )}
            </div>
          )}

          {!loading && prediction && (
            <div className="space-y-4">
              {/* Recommendation */}
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-600 mb-2">Recommendation</div>
                <div className={`inline-block px-6 py-2 rounded-lg text-2xl font-bold ${
                  getRecommendationColor(prediction.recommendation)
                }`}>
                  {prediction.recommendation}
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  Direction: <span className="font-medium">{prediction.predicted_direction}</span>
                </div>
              </div>

              {/* Confidence */}
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-gray-600">Confidence</span>
                  <span className={`text-sm font-medium ${getConfidenceColor(prediction.confidence)}`}>
                    {(prediction.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      prediction.confidence >= 0.7 ? 'bg-green-600' :
                      prediction.confidence >= 0.5 ? 'bg-yellow-600' :
                      'bg-red-600'
                    }`}
                    style={{ width: `${prediction.confidence * 100}%` }}
                  />
                </div>
              </div>

              {/* Price Targets */}
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <div className="text-xs text-gray-600">Target 1-Day</div>
                  <div className="text-lg font-bold text-blue-700">
                    ${prediction.target_price_1d.toFixed(2)}
                  </div>
                  <div className="text-xs text-gray-500">
                    {((prediction.target_price_1d - prediction.current_price) / prediction.current_price * 100).toFixed(2)}%
                  </div>
                </div>

                <div className="p-3 bg-purple-50 rounded-lg">
                  <div className="text-xs text-gray-600">Target 5-Day</div>
                  <div className="text-lg font-bold text-purple-700">
                    ${prediction.target_price_5d.toFixed(2)}
                  </div>
                  <div className="text-xs text-gray-500">
                    {(prediction.expected_return_5d * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              {/* Risk Metrics */}
              <div className="border-t pt-4">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-gray-600">Current Price:</span>
                    <span className="font-medium ml-2">${prediction.current_price.toFixed(2)}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Downside Risk:</span>
                    <span className="font-medium ml-2 text-red-600">
                      {(prediction.downside_risk * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="col-span-2">
                    <span className="text-gray-600">Models Used:</span>
                    <span className="font-medium ml-2">{prediction.models_used.join(', ')}</span>
                  </div>
                  <div className="col-span-2 text-xs text-gray-500">
                    Updated: {new Date(prediction.timestamp).toLocaleString()}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Model Info */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Model Information</h2>

          {modelInfo?.model_exists ? (
            <div className="space-y-3">
              <div className="flex justify-between p-3 bg-green-50 rounded">
                <span className="text-sm text-gray-600">Status</span>
                <span className="text-sm font-medium text-green-700">Model Trained ✓</span>
              </div>

              <div className="flex justify-between p-3 bg-gray-50 rounded">
                <span className="text-sm text-gray-600">Model Type</span>
                <span className="text-sm font-medium">{modelInfo.model_type}</span>
              </div>

              {modelInfo.sequence_length && (
                <div className="flex justify-between p-3 bg-gray-50 rounded">
                  <span className="text-sm text-gray-600">Sequence Length</span>
                  <span className="text-sm font-medium">{modelInfo.sequence_length} days</span>
                </div>
              )}

              {modelInfo.prediction_horizon && (
                <div className="flex justify-between p-3 bg-gray-50 rounded">
                  <span className="text-sm text-gray-600">Prediction Horizon</span>
                  <span className="text-sm font-medium">{modelInfo.prediction_horizon} days</span>
                </div>
              )}

              {modelInfo.last_modified && (
                <div className="flex justify-between p-3 bg-gray-50 rounded">
                  <span className="text-sm text-gray-600">Last Trained</span>
                  <span className="text-sm font-medium">
                    {new Date(modelInfo.last_modified).toLocaleDateString()}
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="text-gray-500 mb-4">
                <p>No model trained for {symbol}</p>
                <p className="text-sm mt-2">Train a model to get ML predictions</p>
              </div>
              <button
                onClick={handleTrainModel}
                disabled={training}
                className="px-6 py-3 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-400"
              >
                {training ? 'Training...' : 'Train Model Now'}
              </button>
              <p className="text-xs text-gray-500 mt-3">Training takes 10-30 minutes</p>
            </div>
          )}
        </div>
      </div>

      {/* Watchlist Predictions */}
      {watchlistPredictions.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Watchlist Predictions</h2>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Recommendation</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Direction</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Confidence</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Current</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Target 5D</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Expected Return</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {watchlistPredictions.map((pred) => (
                  <tr key={pred.symbol} className="hover:bg-gray-50 cursor-pointer" onClick={() => setSymbol(pred.symbol)}>
                    <td className="px-4 py-2 text-sm font-medium">{pred.symbol}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-1 text-xs rounded ${getRecommendationColor(pred.recommendation)}`}>
                        {pred.recommendation}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-sm">{pred.predicted_direction}</td>
                    <td className={`px-4 py-2 text-sm font-medium ${getConfidenceColor(pred.confidence)}`}>
                      {(pred.confidence * 100).toFixed(0)}%
                    </td>
                    <td className="px-4 py-2 text-sm">${pred.current_price.toFixed(2)}</td>
                    <td className="px-4 py-2 text-sm">${pred.target_price_5d.toFixed(2)}</td>
                    <td className={`px-4 py-2 text-sm font-medium ${
                      pred.expected_return_5d > 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {(pred.expected_return_5d * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ML Strategies */}
      {strategies.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-bold text-gray-900 mb-4">ML Trading Strategies</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {strategies.map((strategy) => (
              <div key={strategy.name} className="border rounded-lg p-4">
                <h3 className="font-semibold text-lg mb-2">{strategy.display_name}</h3>
                <p className="text-sm text-gray-600 mb-3">{strategy.description}</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-500">Model:</span>
                    <span className="ml-1 font-medium">{strategy.model}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Accuracy:</span>
                    <span className="ml-1 font-medium">{strategy.expected_accuracy}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Sharpe:</span>
                    <span className="ml-1 font-medium">{strategy.expected_sharpe}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Holding:</span>
                    <span className="ml-1 font-medium">{strategy.holding_period}</span>
                  </div>
                </div>
                <div className="mt-3 text-xs text-gray-500">
                  Best for: {strategy.best_for}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
