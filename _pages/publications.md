---
layout: page
permalink: /publications/
title: Research
description: publications by categories in reversed chronological order.
years: [2026, 2025, 2024]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="publications">

<h1>Published</h1>
{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

<h1>Under Review</h1>
{% bibliography -f papers -q @*[status=underreview]* %}

</div>
