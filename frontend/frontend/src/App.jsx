import React, { useState } from 'react';
import { FileUpload, Loader, Results } from './components';
import { GraduationCap, Upload } from 'lucide-react';

function App() {
  const [answerKey, setAnswerKey] = useState(null);
  const [studentAnswers, setStudentAnswers] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (files, setFileState) => {
    setFileState(files);
  };

  const handleUpload = async () => {
    if (!answerKey?.length || !studentAnswers?.length) {
      alert("Please upload both answer key and student answers.");
      return;
    }

    const formData = new FormData();
    Array.from(answerKey).forEach(file => formData.append("answerKey", file));
    Array.from(studentAnswers).forEach(file => formData.append("studentAnswers", file));

    try {
      setLoading(true);
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Error uploading files:", error);
      alert("An error occurred while processing the files.");
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setAnswerKey(null);
    setStudentAnswers(null);
    setResults(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 relative overflow-hidden">
      {/* Decorative background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-16 -right-16 w-32 h-32 bg-blue-200 rounded-full opacity-20" />
        <div className="absolute top-1/2 -left-8 w-24 h-24 bg-indigo-200 rounded-full opacity-20" />
        <div className="absolute -bottom-8 right-1/2 w-40 h-40 bg-purple-200 rounded-full opacity-20" />
      </div>

      <div className="container mx-auto px-4 py-8 relative">
        <header className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-2">
            <GraduationCap className="w-8 h-8 text-indigo-600" />
            <h1 className="text-4xl font-bold text-gray-800 font-display">
              Image-Based Answer Scoring System
            </h1>
          </div>
          <p className="text-gray-600 mt-2">
            Upload answer keys and student responses for automated scoring
          </p>
        </header>

        {loading ? (
          <Loader />
        ) : results ? (
          <Results results={results} onReset={resetForm} />
        ) : (
          <FileUpload
            answerKey={answerKey}
            studentAnswers={studentAnswers}
            onAnswerKeyChange={(files) => handleFileChange(files, setAnswerKey)}
            onStudentAnswersChange={(files) => handleFileChange(files, setStudentAnswers)}
            onSubmit={handleUpload}
          />
        )}
      </div>
    </div>
  );
}

export default App;
