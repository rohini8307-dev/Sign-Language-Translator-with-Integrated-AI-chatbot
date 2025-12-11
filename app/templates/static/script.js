AOS.init({duration:800,once:true});

const STREAMLIT_URL = 'https://your-streamlit-app-url';

document.getElementById('tryBtn').addEventListener('click', ()=>{
  window.open(STREAMLIT_URL, '_blank');
});
document.getElementById('launchStreamlit').addEventListener('click', ()=>{
  window.open(STREAMLIT_URL, '_blank');
});

gsap.from('.logo', {y:-8, opacity:0, duration:0.8});
gsap.from('.title', {y:18, opacity:0, duration:0.9, delay:0.2});
gsap.from('.demo-preview', {scale:0.98, opacity:0, duration:0.9, delay:0.3});

try{
  VANTA.NET({
    el: document.querySelector('.hero'),
    color: 0x00e6ff,
    backgroundColor: 0x05070a,
    points: 10.00,
    maxDistance: 22.00,
    spacing: 18.00
  });
}catch(e){console.warn('Vanta failed',e)}
