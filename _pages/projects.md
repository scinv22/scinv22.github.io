---
layout: page
title: Life
permalink: /projects/
description: "\"The object of life is not to be on the side of the majority, but to escape finding oneself in the ranks of the insane\" - Marcus Aurelius"
nav: true
nav_order: 2
---

<!-- pages/projects.md -->
<div class="gallery-page">

  <div class="gallery-filters">
    <button class="gallery-filter-btn active" data-filter="all" type="button">All</button>
    <button class="gallery-filter-btn" data-filter="art" type="button">Art</button>
    <button class="gallery-filter-btn" data-filter="photography" type="button">Photography</button>
  </div>

  <div class="masonry-gallery">

    <div class="masonry-item" data-category="art">
      <div class="masonry-card">
        {% include figure.html path="assets/img/p1.jpg" alt="Satyajit Ray - sketch" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Ray</h3>
          <span class="masonry-tag">Sketch</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/j2.jpg" alt="Ethereal view of the Himalayas from hostel-1" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Ethereal Himalayas</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="art">
      <div class="masonry-card">
        {% include figure.html path="assets/img/p2.jpg" alt="Tiger - sketch" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Aura</h3>
          <span class="masonry-tag">Sketch</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/k2.jpg" alt="Birds over a Tibetan Buddhist monastery" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Wings Over the Monastery</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="art">
      <div class="masonry-card">
        {% include figure.html path="assets/img/p3.jpg" alt="Wolf - sketch" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Lone Wolf</h3>
          <span class="masonry-tag">Sketch</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/a2.jpg" alt="Garh Panchakot" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Panchakot</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/a4.jpg" alt="Sunset over Deul" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">7th of December</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="art">
      <div class="masonry-card">
        {% include figure.html path="assets/img/p4.jpg" alt="Cat — A sketch" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Ethereal</h3>
          <span class="masonry-tag">Sketch</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/k3.jpg" alt="Golden spire of a Tibetan Buddhist monastery in Kalimpong" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Golden Spire</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/a1.jpg" alt="Sunset over Mahabalipuram beach" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Chowdhurani at Sundown</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/k4.jpg" alt="Prayer wheels at a Tibetan Buddhist monastery" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Prayer Wheels</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/k1.jpg" alt="Misty hills of Kalimpong" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Misty Hills of Kalimpong</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="art">
      <div class="masonry-card">
        {% include figure.html path="assets/img/p1.jpg" alt="Einstein - Painting" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Genius</h3>
          <span class="masonry-tag">Painting</span>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
  document.addEventListener('DOMContentLoaded', function () {
    var buttons = document.querySelectorAll('.gallery-filter-btn');
    var items = document.querySelectorAll('.masonry-item');

    buttons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        buttons.forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');

        var filter = btn.getAttribute('data-filter');
        items.forEach(function (item) {
          var show = filter === 'all' || item.getAttribute('data-category') === filter;
          item.style.display = show ? '' : 'none';
        });
      });
    });
  });
</script>
