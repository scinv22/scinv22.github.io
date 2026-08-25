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
        {% include figure.html path="assets/img/p1.jpg" alt="Einstein - ink portrait sketch" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Einstein</h3>
          <span class="masonry-tag">Ink Sketch</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/j2.jpg" alt="Ethereal view of the Himalayas from the hostel" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Ethereal Himalayas</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="art">
      <div class="masonry-card">
        {% include figure.html path="assets/img/p2.jpg" alt="The lioness - watercolor painting" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">The Lioness</h3>
          <span class="masonry-tag">Watercolor</span>
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
        {% include figure.html path="assets/img/p3.jpg" alt="Curious - watercolor portrait" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Curious</h3>
          <span class="masonry-tag">Watercolor</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/a2.jpg" alt="A lotus flower in search of beauty" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">In Search of Beauty</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/4.jpg" alt="Sunset over the ocean waves" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Where the Waves Break</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="art">
      <div class="masonry-card">
        {% include figure.html path="assets/img/p4.jpg" alt="Terrace view - watercolor painting" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Terrace View</h3>
          <span class="masonry-tag">Watercolor</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/k3.jpg" alt="Golden spire of a Tibetan Buddhist monastery" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Golden Spire</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography">
      <div class="masonry-card">
        {% include figure.html path="assets/img/a1.jpg" alt="Looking up through the branches of a tree" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Reaching Skyward</h3>
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
