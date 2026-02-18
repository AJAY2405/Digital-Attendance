// camera_mark.js
const startMarkBtn = document.getElementById("startMarkBtn");
const stopMarkBtn = document.getElementById("stopMarkBtn");
const markVideo = document.getElementById("markVideo");
const markStatus = document.getElementById("markStatus");
const recognizedList = document.getElementById("recognizedList");
const toastContainer = document.getElementById("toastContainer");

let markStream = null;
let markInterval = null;
let recognizedIds = new Set();

// Format date/time as "DD Mon HH:MM AM/PM"
function formatDateTime(date) {
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const day = String(date.getDate()).padStart(2, '0');
  const month = months[date.getMonth()];
  const hours = date.getHours();
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const ampm = hours >= 12 ? 'PM' : 'AM';
  const displayHours = hours % 12 || 12;
  return `${day} ${month} ${displayHours}:${minutes} ${ampm}`;
}

// Show toast notification
function showToast(message, type = 'info', duration = 3000) {
  const toast = document.createElement('div');
  toast.className = `toast p-3 toast-${type}`;
  toast.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <span>${message}</span>
      <button type="button" class="btn-close btn-close-white ms-2" onclick="this.parentElement.parentElement.remove()" style="font-size: 0.8rem;"></button>
    </div>
  `;
  
  toastContainer.appendChild(toast);
  
  // Auto-remove after duration
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

startMarkBtn.addEventListener("click", async () => {
  startMarkBtn.disabled = true;
  stopMarkBtn.disabled = false;
  try {
    markStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    markVideo.srcObject = markStream;
    await markVideo.play();
    markStatus.innerText = "Scanning...";
    markInterval = setInterval(captureAndRecognize, 1200);
  } catch (err) {
    alert("Camera error: " + err.message);
    startMarkBtn.disabled = false;
    stopMarkBtn.disabled = true;
  }
});

stopMarkBtn.addEventListener("click", () => {
  if (markInterval) clearInterval(markInterval);
  if (markStream) markStream.getTracks().forEach(t => t.stop());
  startMarkBtn.disabled = false;
  stopMarkBtn.disabled = true;
  markStatus.innerText = "Stopped";
});

async function captureAndRecognize() {
  const canvas = document.createElement("canvas");
  canvas.width = markVideo.videoWidth || 640;
  canvas.height = markVideo.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(markVideo, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.85));
  const fd = new FormData();
  fd.append("image", blob, "snap.jpg");
  try {
    const res = await fetch("/recognize_face", { method: "POST", body: fd });
    const j = await res.json();
    if (j.recognized) {
      if (j.status === "marked") {
        // First time marked today - SUCCESS
        markStatus.innerText = `✓ Marked: ${j.name} (conf ${Math.round(j.confidence*100)}%)`;
        markStatus.style.color = "green";
        if (!recognizedIds.has(j.student_id)) {
          recognizedIds.add(j.student_id);
          const li = document.createElement("li");
          li.className = "list-group-item list-group-item-success";
          li.innerText = `✓ ${j.name} — ${formatDateTime(new Date())}`;
          recognizedList.prepend(li);
          // Show success popup
          showToast(`✓ Attendance Marked: ${j.name}`, 'success', 3000);
        }
      } else if (j.status === "already_marked") {
        // Already marked today - WARNING
        markStatus.innerText = `Already marked: ${j.name}`;
        markStatus.style.color = "orange";
        // Show warning popup
        showToast(`⚠️ ${j.name}'s attendance already marked today`, 'warning', 4000);
      }
    } else {
      if (j.error) {
        markStatus.innerText = `✗ Not recognized: ${j.error}`;
        showToast(`✗ ${j.error}`, 'error', 2000);
      } else {
        markStatus.innerText = `✗ Not recognized`;
        showToast(`✗ Face not recognized`, 'error', 2000);
      }
      markStatus.style.color = "red";
    }
  } catch (err) {
    console.error(err);
    markStatus.innerText = "Error: " + err.message;
    markStatus.style.color = "red";
    showToast(`✗ Error: ${err.message}`, 'error', 3000);
  }
}