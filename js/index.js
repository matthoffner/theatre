import React from "https://cdn.skypack.dev/react";
import ReactDOM from "https://cdn.skypack.dev/react-dom";
import htm from "https://cdn.skypack.dev/htm";

window.html = htm.bind(React.createElement);
const evtSource = new EventSource("http://127.0.0.1:8000/stream");

const Theatre = () => {
  const [messages, setMessage] = React.useState([]);
  React.useEffect(() => {
    evtSource.addEventListener("new_message", async function (event) {
      // Logic to handle status updates
      setMessage((messages) => [...messages, event.data]);
    });

    evtSource.addEventListener("end_event", function (event) {
      setMessage((messages) => [...messages, event.data]);
      evtSource.close();
    });

    return () => {
      evtSource.close();
    };
  }, []);

  console.log(messages);

  return html`<div style=${{
    backgroundColor: "white",
    color: "black",
    textAlign: "left",
    padding: "20px",
    borderRadius: "5px",
    minHeight: "100vh",
    border: "20px solid",
    borderStyle: 'solid',
    borderImage: "linear-gradient(to top, red, black) 1",
    fontFamily: "Arial",
    transition: "border-width 0.6s linear"
  }}>
    ${messages.map(message => html`<div>${message}</div>`)}
  </div>`;
};

ReactDOM.render(html`<${Theatre} />`, document.getElementById("root"));
