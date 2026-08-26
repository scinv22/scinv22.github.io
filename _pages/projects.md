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
    <button class="gallery-filter-btn" data-filter="travel" type="button">Travel</button>
  </div>

  <div class="gallery-subfilters" data-parent="art">
    <button class="gallery-subfilter-btn active" data-subfilter="all" type="button">All</button>
    <button class="gallery-subfilter-btn" data-subfilter="sketches" type="button">Sketches</button>
    <button class="gallery-subfilter-btn" data-subfilter="paintings" type="button">Paintings</button>
  </div>

  <div class="gallery-subfilters" data-parent="photography">
    <button class="gallery-subfilter-btn active" data-subfilter="all" type="button">All</button>
    <button class="gallery-subfilter-btn" data-subfilter="yin" type="button">Yin</button>
    <button class="gallery-subfilter-btn" data-subfilter="yang" type="button">Yang</button>
  </div>

  <div class="gallery-subfilters" data-parent="travel">
    <button class="gallery-subfilter-btn active" data-subfilter="all" type="button">All</button>
    <button class="gallery-subfilter-btn" data-subfilter="kalimpong" type="button">Kalimpong</button>
    <button class="gallery-subfilter-btn" data-subfilter="jodhpur" type="button">Jodhpur</button>
    <button class="gallery-subfilter-btn" data-subfilter="delhi" type="button">Delhi</button>
    <button class="gallery-subfilter-btn" data-subfilter="bhubaneshwar" type="button">Bhubaneshwar</button>
    <button class="gallery-subfilter-btn" data-subfilter="chennai" type="button">Chennai</button>
    <button class="gallery-subfilter-btn" data-subfilter="mahabalipuram" type="button">Mahabalipuram</button>
    <button class="gallery-subfilter-btn" data-subfilter="bishnupur" type="button">Bishnupur</button>
    <button class="gallery-subfilter-btn" data-subfilter="garh-panchakot" type="button">Garh Panchakot</button>
    <button class="gallery-subfilter-btn" data-subfilter="ajodhya" type="button">Ajodhya</button>
    <button class="gallery-subfilter-btn" data-subfilter="durgapur" type="button">Durgapur</button>
    <button class="gallery-subfilter-btn" data-subfilter="parasnath" type="button">Parasnath</button>
    <button class="gallery-subfilter-btn" data-subfilter="shantiniketan" type="button">Shantiniketan</button>
  </div>

  <div class="masonry-gallery">

    <div class="masonry-item" data-category="art" data-subcategory="sketches">
      <div class="masonry-card">
        {% include figure.html path="assets/img/p1.jpg" alt="Satyajit Ray - sketch" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Ray</h3>
          <span class="masonry-tag">Sketch</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography" data-subcategory="yin">
      <div class="masonry-card">
        {% include figure.html path="assets/img/a5.jpg" alt="" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">सफ़र</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography" data-subcategory="yin">
      <div class="masonry-card">
        {% include figure.html path="assets/img/a7.jpg" alt=" Incoming storm and darkness" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Ominous</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

     <div class="masonry-item" data-category="photography" data-subcategory="yang">
      <div class="masonry-card">
        {% include figure.html path="assets/img/a8.jpg" alt="Sunset" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Golden Hour</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

     <div class="masonry-item" data-category="photography" data-subcategory="yin">
      <div class="masonry-card">
        {% include figure.html path="assets/img/a6.jpg" alt="Abandoned" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Abandoned</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="art" data-subcategory="sketches">
      <div class="masonry-card">
        {% include figure.html path="assets/img/p2.jpg" alt="Tiger - sketch" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Aura</h3>
          <span class="masonry-tag">Sketch</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="travel" data-subcategory="kalimpong">
      <div class="masonry-card">
        {% include figure.html path="assets/img/k2.jpg" alt="Birds over a Tibetan Buddhist monastery" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Wings Over the Monastery</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="art" data-subcategory="sketches">
      <div class="masonry-card">
        {% include figure.html path="assets/img/p3.jpg" alt="Wolf - sketch" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Lone Wolf</h3>
          <span class="masonry-tag">Sketch</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="travel" data-subcategory="garh-panchakot">
      <div class="masonry-card">
        {% include figure.html path="assets/img/a2.jpg" alt="Garh Panchakot" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Panchakot</h3>
          <span class="masonry-tag">Garh Panchakot</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography" data-subcategory="yang">
      <div class="masonry-card">
        {% include figure.html path="assets/img/a4.jpg" alt="Sunset over Deul" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">7th of December</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="art" data-subcategory="sketches">
      <div class="masonry-card">
        {% include figure.html path="assets/img/p4.jpg" alt="Cat — A sketch" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Ethereal</h3>
          <span class="masonry-tag">Sketch</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="travel" data-subcategory="kalimpong">
      <div class="masonry-card">
        {% include figure.html path="assets/img/k3.jpg" alt="Golden spire of a Tibetan Buddhist monastery in Kalimpong" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Golden Spire</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="travel" data-subcategory="kalimpong">
      <div class="masonry-card">
        {% include figure.html path="assets/img/k4.jpg" alt="Prayer wheels at a Tibetan Buddhist monastery" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Prayer Wheels</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="travel" data-subcategory="kalimpong">
      <div class="masonry-card">
        {% include figure.html path="assets/img/k1.jpg" alt="Misty hills of Kalimpong" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Misty Hills of Kalimpong</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="art" data-subcategory="paintings">
      <div class="masonry-card">
        {% include figure.html path="assets/img/p5.jpg" alt="Einstein - Painting" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Genius</h3>
          <span class="masonry-tag">Painting</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography" data-subcategory="yin">
      <div class="masonry-card">
        {% include figure.html path="assets/img/d1.jpg" alt="Mood" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Moody</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    <div class="masonry-item" data-category="photography" data-subcategory="yin">
      <div class="masonry-card">
        {% include figure.html path="assets/img/d2.jpg" alt="Depths of Darkness" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Depth</h3>
          <span class="masonry-tag">Photography</span>
        </div>
      </div>
    </div>

    
    <div class="masonry-item" data-category="travel" data-subcategory="mahabalipuram">
      <div class="masonry-card">
        {% include figure.html path="assets/img/m0.jpg" alt="Sunset over Mahabalipuram beach" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Chowdhurani at Sundown</h3>
          <span class="masonry-tag">Mahabalipuram</span>
        </div>
      </div>
    </div>

    
    <div class="masonry-item" data-category="travel" data-subcategory="mahabalipuram">
      <div class="masonry-card">
        {% include figure.html path="assets/img/m1.jpg" alt="Rock-cut" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Rock-cut</h3>
          <span class="masonry-tag">Mahabalipuram</span>
        </div>
      </div>
    </div>

    
    <div class="masonry-item" data-category="travel" data-subcategory="mahabalipuram">
      <div class="masonry-card">
        {% include figure.html path="assets/img/m2.jpg" alt="Shore Temple" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Shore Temple</h3>
          <span class="masonry-tag">Mahabalipuram</span>
        </div>
      </div>
    </div>

    
    <div class="masonry-item" data-category="travel" data-subcategory="mahabalipuram">
      <div class="masonry-card">
        {% include figure.html path="assets/img/m3.jpg" alt="Ganesha Ratha" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Ganesha Ratha</h3>
          <span class="masonry-tag">Mahabalipuram</span>
        </div>
      </div>
    </div>

    
    <div class="masonry-item" data-category="travel" data-subcategory="mahabalipuram">
      <div class="masonry-card">
        {% include figure.html path="assets/img/m4.jpg" alt="Rock-cut statue" zoomable=true %}
        <div class="masonry-overlay">
          <h3 class="masonry-title">Daemon</h3>
          <span class="masonry-tag">Mahabalipuram</span>
        </div>
      </div>
    </div>

  </div>

  <p class="gallery-empty" id="galleryEmpty" hidden>This corner of the gallery is still being curated — check back soon.</p>
</div>

<script>
  document.addEventListener('DOMContentLoaded', function () {
    var mainButtons = document.querySelectorAll('.gallery-filter-btn');
    var subGroups = document.querySelectorAll('.gallery-subfilters');
    var items = document.querySelectorAll('.masonry-item');
    var emptyMessage = document.getElementById('galleryEmpty');

    function applyFilter(category, subcategory) {
      var visibleCount = 0;
      items.forEach(function (item) {
        var matchesCategory = category === 'all' || item.getAttribute('data-category') === category;
        var matchesSub = !subcategory || subcategory === 'all' || item.getAttribute('data-subcategory') === subcategory;
        var show = matchesCategory && matchesSub;
        item.style.display = show ? '' : 'none';
        if (show) { visibleCount++; }
      });
      if (emptyMessage) { emptyMessage.hidden = visibleCount !== 0; }
    }

    function showSubgroup(category) {
      subGroups.forEach(function (group) {
        var isMatch = group.getAttribute('data-parent') === category;
        group.classList.toggle('is-open', isMatch);
        if (isMatch) {
          group.querySelectorAll('.gallery-subfilter-btn').forEach(function (b) {
            b.classList.toggle('active', b.getAttribute('data-subfilter') === 'all');
          });
        }
      });
    }

    mainButtons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        mainButtons.forEach(function (b) { b.classList.remove('active'); });
        btn.classList.add('active');

        var category = btn.getAttribute('data-filter');
        showSubgroup(category);
        applyFilter(category, 'all');
      });
    });

    subGroups.forEach(function (group) {
      group.querySelectorAll('.gallery-subfilter-btn').forEach(function (subBtn) {
        subBtn.addEventListener('click', function () {
          group.querySelectorAll('.gallery-subfilter-btn').forEach(function (b) { b.classList.remove('active'); });
          subBtn.classList.add('active');
          applyFilter(group.getAttribute('data-parent'), subBtn.getAttribute('data-subfilter'));
        });
      });
    });
  });
</script>
