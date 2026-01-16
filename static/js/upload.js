window.onerror = () => true;


const analyzeBtn = document.querySelector("#analyzeBtn");
const fileInput = document.querySelector("#mriInput");
const viewer = document.querySelector("#viewer");

analyzeBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) {
    alert("Please upload MRI images first");
    return;
  }

  const formData = new FormData();

  // append all slices
  for (let file of fileInput.files) {
    formData.append("files", file);
  }

  viewer.innerHTML = "⏳ Analyzing MRI series...";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    viewer.innerHTML = `
      <h3>Prediction: ${data.label}</h3>
      <img src="data:image/png;base64,${data.gradcam}" style="width:100%;border-radius:12px">
    `;
  } catch (err) {
    viewer.innerHTML = "❌ Analysis failed";
    console.error(err);
  }
});


document.getElementById("analyzeBtn").onclick = async () => {
  const file = document.getElementById("fileInput").files[0];
  if (!file) return alert("Upload MRI image");

  const fd = new FormData();
  fd.append("image", file);

  const res = await fetch("/predict", { method: "POST", body: fd });
  const data = await res.json();

  document.getElementById("resultBox").innerHTML =
    `Diagnosis: <b>${data.class}</b> (${data.confidence})`;
};
