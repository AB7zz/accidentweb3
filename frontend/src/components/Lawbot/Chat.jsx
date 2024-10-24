import React, { useState,useEffect } from 'react';



function ChatApp() {
    const [messages, setMessages] = useState([])
    
    // const configuration = new Configuration({
    //     apiKey: process.env.OPENAI_API_KEY,
    // });
    // const openai = new OpenAIApi(configuration);
    // const openaiClient = new openai.OpenAIApi({
    //     apiKey: "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",
    //   });
    
    // const openai = new OpenAI({
    //     apiKey: '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',
    //     dangerouslyAllowBrowser: true,
    //   });
    // //   const openai = new OpenAIApi(configuration);
    useEffect(() => {
      console.log('Messages:', messages);
    }, [messages]);

    const handleMessageSubmit = async (messageText) => {
      try {
        const newMessage = {
          text: messageText,
          sender: 'user',
          timestamp: 'Just now',
        };
        const updatedMessages = [...messages, newMessage];
    
        // Add the user's message to the state first
        setMessages(updatedMessages);
    
<<<<<<< HEAD
        const apiUrl = 'http://localhost:3000/bot';
=======
        const apiUrl = 'https://hackcdent-891d0b31e96e.herokuapp.com/bot';
>>>>>>> b4f3eb85d67d992a13adc02dd6915f499977161f
    
        const response = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: messageText }),
        });
    
        if (response.ok) {
          // Get the response data
          const responseData = await response.json();
          console.log(responseData);
          // Create a message from the response
          const responseMessage = {
            text: responseData.output,
            sender: 'server', // You can use 'server' or another identifier
            timestamp: 'Just now',
          };
    
          // Add the server's response message to the state using the updatedMessages variable
          const updatedMessagesWithResponse = [...updatedMessages, responseMessage];
          setMessages(updatedMessagesWithResponse);
          console.log(updatedMessagesWithResponse);
        } else {
          console.error('Error sending message to backend.');
        }
      } catch (error) {
        console.error('Error making API request:', error);
      }
    };
    

  return (
    <div className="flex flex-col  h-full w-full bg-white shadow-xl rounded-lg border border-gray-300">
      {/* Display messages here */}
      <div className="flex flex-col flex-grow h-0 p-4 overflow-y-auto">
        {messages && messages.map((message, index) => (
          <div
            key={index}
            className={`flex w-full mt-2 space-x-3 max-w-xs ${
              message.sender === 'user' ? 'ml-auto justify-end' : ''
            }`}
          >
            <div>
              <div
                className={`${
                  message.sender === 'user' ? 'bg-blue-600 text-white rounded-l-lg rounded-br-lg' : 'bg-gray-300 rounded-r-lg rounded-bl-lg'
                } p-3`}
              >
                <p className="text-sm">{message.text}</p>
              </div>
              {/* <span className="text-xs text-gray-500 leading-none">{message.timestamp}</span> */}
            </div>
            {/* {message.sender !== 'user' && (
              <div className="flex-shrink-0 h-10 w-10 rounded-full bg-gray-300"></div>
            )} */}
          </div>
        ))}
      </div>

      {/* Input for sending new messages */}
      <div className="border border-gray-300 p-4 rounded-lg">
        <input
          className="flex items-center h-10 w-full rounded px-3 text-sm focus:outline-none"
          type="text"
          placeholder="Type your message…"
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              handleMessageSubmit(e.target.value);
              e.target.value = '';
            }
          }}
        />
      </div>
    </div>
  )
        }

export default ChatApp;
