import React from 'react';
import './TranslationBox.css';

function TranslationBox({ translation }) {
    return (
        <div className="translation-box">
            <h2>Translation</h2>
            <p className={`translation-text ${!translation ? 'empty' : ''}`}>
                {translation || 'Waiting for translation...'}
            </p>
        </div>
    );
}

export default TranslationBox;
