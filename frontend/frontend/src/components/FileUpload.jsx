import React from 'react';
import { Upload, File } from 'lucide-react';

const FileUpload = ({
  answerKey,
  studentAnswers,
  onAnswerKeyChange,
  onStudentAnswersChange,
  onSubmit,
}) => {
  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white rounded-2xl shadow-xl p-8 space-y-8">
        <div className="space-y-6">
          <div className="upload-section">
            <label className="block text-lg font-semibold text-gray-700 mb-2">
              Upload Answer Key Images
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 transition-colors hover:border-indigo-500">
              <div className="flex flex-col items-center">
                <Upload className="w-12 h-12 text-gray-400 mb-4" />
                <input
                  type="file"
                  multiple
                  onChange={(e) => onAnswerKeyChange(e.target.files)}
                  className="hidden"
                  id="answer-key-input"
                  accept="image/*"
                />
                <label
                  htmlFor="answer-key-input"
                  className="cursor-pointer text-indigo-600 hover:text-indigo-500"
                >
                  Choose files
                </label>
                {answerKey && answerKey.length > 0 && (
                  <div className="mt-4 text-sm text-gray-500">
                    {Array.from(answerKey).map((file, index) => (
                      <div key={index} className="flex items-center gap-2">
                        <File className="w-4 h-4" />
                        {file.name}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="upload-section">
            <label className="block text-lg font-semibold text-gray-700 mb-2">
              Upload Student Answer Images
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 transition-colors hover:border-indigo-500">
              <div className="flex flex-col items-center">
                <Upload className="w-12 h-12 text-gray-400 mb-4" />
                <input
                  type="file"
                  multiple
                  onChange={(e) => onStudentAnswersChange(e.target.files)}
                  className="hidden"
                  id="student-answers-input"
                  accept="image/*"
                />
                <label
                  htmlFor="student-answers-input"
                  className="cursor-pointer text-indigo-600 hover:text-indigo-500"
                >
                  Choose files
                </label>
                {studentAnswers && studentAnswers.length > 0 && (
                  <div className="mt-4 text-sm text-gray-500">
                    {Array.from(studentAnswers).map((file, index) => (
                      <div key={index} className="flex items-center gap-2">
                        <File className="w-4 h-4" />
                        {file.name}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <button
            onClick={onSubmit}
            className="px-8 py-3 bg-indigo-600 text-white rounded-lg shadow-md hover:bg-indigo-500 transition"
          >
            Submit
          </button>
        </div>
      </div>
    </div>
  );
};

export default FileUpload;
