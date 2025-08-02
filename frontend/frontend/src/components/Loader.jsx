import React from 'react';

const Loader = () => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white p-8 rounded-2xl shadow-2xl flex flex-col items-center max-w-sm mx-4">
        <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-indigo-600 mb-4"></div>
        <h3 className="text-xl font-semibold text-gray-800 mb-2">Processing your results...</h3>
        <p className="text-gray-600 text-center">Please wait while we analyze your answersheet</p>
      </div>
    </div>
  );
};

export default Loader;
