'use strict';

document.addEventListener('DOMContentLoaded', () => {

  // Adding event listeners to movie cards
  const movieCardsElements = document.querySelectorAll('[data-movie-card]');
  const backTop = document.querySelector('[data-back-top]');

  let activeCard = null;

  movieCardsElements.forEach(movieCard => {
    movieCard.addEventListener('click', () => {
      const reviewCard = movieCard.querySelector('[data-movie-review-card]');
      if (reviewCard !== activeCard) {
        if (activeCard) {
          activeCard.classList.remove('active');
        }
        reviewCard.classList.add('active');
        activeCard = reviewCard;
      }
    });

    const closeBtn = movieCard.querySelector('[data-close]');
    closeBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      const reviewCard = movieCard.querySelector('[data-movie-review-card]');
      reviewCard.classList.remove('active');
      activeCard = null;
    });
  });

  const closeBtnFlash = document.querySelector('[data-close-flash]');
  const flashCard = document.querySelector('[data-flash');

  closeBtnFlash.addEventListener('click',function(){
    flashCard.style.display = "none";
  });

  window.addEventListener("scroll", function(){
    if(window.scrollY > 100){
       backTop.classList.add('active');
    }
    else{
      backTop.classList.remove('active');
    }
  })

});