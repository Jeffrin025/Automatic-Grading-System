import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { RefreshCw, Award, Target } from 'lucide-react';

const Results = ({ results, onReset }) => {
  const chartData = Object.entries(results.detailed_scores).map(([question, data]) => ({
    name: `Q${question}`,
    score: data.score,
    maxScore: data.max_score,
  }));

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-fadeIn">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-xl shadow-lg p-6 flex items-center space-x-4">
          <div className="p-3 bg-green-100 rounded-full">
            <Target className="w-6 h-6 text-green-600" />
          </div>
          <div>
            <p className="text-sm text-gray-500">Total Score</p>
            <p className="text-2xl font-bold text-gray-800">
              {results.total_score}/{results.total_marks}
            </p>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6 flex items-center space-x-4">
          <div className="p-3 bg-blue-100 rounded-full">
            <Award className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <p className="text-sm text-gray-500">Percentage</p>
            <p className="text-2xl font-bold text-gray-800">{results.percentage}%</p>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Question-wise Scores</h3>
          <div className="space-y-2">
            {Object.entries(results.detailed_scores).map(([question, data]) => (
              <div key={question} className="flex justify-between items-center">
                <span className="text-gray-600">Question {question}</span>
                <span className="font-medium black-gray-100">
                  {data.score}/{data.max_score}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Score Distribution</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="score" fill="#4F46E5" />
              <Bar dataKey="maxScore" fill="#E5E7EB" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {results.overall_feedback && (
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Overall Feedback</h3>
          <p className="text-gray-700 text-base">{results.overall_feedback}</p>
        </div>
      )}

      <div className="flex justify-center">
        <button
          onClick={onReset}
          className="flex items-center gap-2 px-6 py-3 bg-gray-800 text-white rounded-lg font-semibold shadow-lg hover:bg-gray-700 transform transition-all duration-200 hover:scale-105"
        >
          <RefreshCw className="w-5 h-5" />
          Score Another Response
        </button>
      </div>
    </div>
  );
};

export default Results;
