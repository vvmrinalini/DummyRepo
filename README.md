<!DOCTYPE html>
<html lang="en" dir="ltr">
  <head>
    <meta charset="utf-8">  
    <title>ChatBot</title>
    <link rel="stylesheet" href="style.css">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@48,400,0,0" />
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@48,400,1,0" />
    <link rel="shortcut icon" href="favicon.ico" type="image/x-icon">
    <script src="script.js" defer></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #E3F2FD;
            margin: 0;
            padding: 0;
        }
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            background-color: white;
            border-bottom: 1px solid #ccc;
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        .navbar-logo img {
            width: 50px; /* Reduced size */
            height: auto;
        }
        .navbar-links {
            display: flex;
            gap: 30px;
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
        }
        .navbar-links a {
            text-decoration: none;
            color: black;
            font-size: 16px;
        }
        .navbar-links a:hover {
            color: green; /* Change to green on hover */
        }
        .content {
            display: flex;
            justify-content: center;
            align-items: center;
            height: calc(100vh - 60px); /* Adjust height to fill the remaining space */
            background: url('img2.jpg') no-repeat center center fixed; /* Updated image source */
            background-size: cover; /* Make the image cover the entire page */
        }
    </style>
  </head>
  <body>
    <nav>
      <div class="navbar">
        <div class="navbar-logo"><img src="fis-logo.png" alt="Logo"></div>
        <div class="navbar-links">
          <a href="/home">Home</a>
          <a href="/form">Raise a Ticket</a>
          <a href="/tickets">My Tickets</a>
        </div>
      </div>
    </nav>
    <div class="content"></div>
    <button class="chatbot-toggler">
      <span class="material-symbols-rounded">mode_comment</span>
      <span class="material-symbols-outlined">close</span>
    </button>
    <div class="chatbot">
      <header>
        <h2>ChatBot</h2>
        <span class="close-btn material-symbols-outlined">close</span>
      </header>
      <ul class="chatbox">
        <li class="chat incoming">
          <span class="material-symbols-outlined">smart_toy</span>
          <p>Hi there 👋<br>How can I help you today?</p>
        </li>
      </ul>
      <div class="chat-input">
        <textarea placeholder="Enter a message..." spellcheck="false" required></textarea>
        <span id="send-btn" class="material-symbols-rounded">send</span>
      </div>
    </div>
  </body>
</html>
