import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import TranslationBox from './TranslationBox.jsx';

const socket = io('http://localhost:5000');

function App() {
    const [translation, setTranslation] = useState('');

    useEffect(() => {
        socket.on('update', (data) => {
            if (data.translation) {
                setTranslation(data.translation);
            }
        });
        return () => {
            socket.off('update');
        };
    }, []);

    return (
        <div className="app">
            <h1>Sign Language Translation</h1>
            <TranslationBox translation={translation} />
        </div>
    );
}

export default App;
