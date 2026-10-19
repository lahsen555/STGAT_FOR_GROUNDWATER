// Author-supplied corrected_v1 exporter, packaged 2026-10-19.
// Set this to the asset created by uploading gee/wells.csv. See gee/README.md.
// Changes for release: configurable asset, remove duplicate declaration,
// normalize the identifier to Well_ID. Scientific calculations are unchanged.
var WELL_ASSET_ID = 'projects/YOUR_PROJECT/assets/wells_gee_houze';
var wells = ee.FeatureCollection(WELL_ASSET_ID).map(function(f) {
  var id = ee.Algorithms.If(f.propertyNames().contains('Well_ID'),
    f.get('Well_ID'), f.get('well_id'));
  return ee.Feature(f.geometry(), {Well_ID: id});
});

var start = ee.Date('2006-01-01');
var end = ee.Date('2023-01-01'); // exclusive
var months = ee.List.sequence(0, end.difference(start, 'month').subtract(1));
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY');
var mod16 = ee.ImageCollection('MODIS/061/MOD16A2GF');
var mod11 = ee.ImageCollection('MODIS/061/MOD11A2');
var mod13 = ee.ImageCollection('MODIS/061/MOD13A2');
var wapor = ee.ImageCollection('FAO/WAPOR/2/L1_AETI_D');
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR');
var tc = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE');
var gldas = ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H');
var srtm = ee.Image('USGS/SRTMGL1_003');
var terrain = srtm.select('elevation').rename('Elevation_m')
  .addBands(ee.Terrain.slope(srtm).rename('Slope_deg')).toDouble();

function maskedBand(name) {
  return ee.Image.constant(0).toDouble().rename(name).updateMask(ee.Image.constant(0));
}
function safeMean(collection, band, name) {
  var avg = collection.select([band]).mean()
    .addBands(ee.Image(0).rename(band).selfMask(), null, false);
  return avg.select([band]).rename(name);
}
function modisPeriod(img, days) {
  var d0 = ee.Date(img.get('system:time_start'));
  var nextYear = ee.Date.fromYMD(ee.Number(d0.get('year')).add(1), 1, 1);
  var d1 = ee.Date(ee.Number(d0.advance(days, 'day').millis()).min(nextYear.millis()));
  return ee.List([d0.millis(), d1.millis()]);
}
function dekadPeriod(img) {
  var d0 = ee.Date(img.get('system:time_start'));
  var day = ee.Number(d0.get('day'));
  var d1 = ee.Date(ee.Algorithms.If(day.lte(10),
    ee.Date.fromYMD(d0.get('year'), d0.get('month'), 11),
    ee.Algorithms.If(day.lte(20),
      ee.Date.fromYMD(d0.get('year'), d0.get('month'), 21),
      ee.Date.fromYMD(d0.get('year'), d0.get('month'), 1).advance(1, 'month'))));
  return ee.List([d0.millis(), d1.millis()]);
}
function overlapDays(period, d0, d1) {
  var a = ee.Number(period.get(0)).max(d0.millis());
  var b = ee.Number(period.get(1)).min(d1.millis());
  return b.subtract(a).divide(86400000).max(0);
}
function weightedComposite(coll, d0, d1, periodFn, preprocess, isTotal, isMean, name) {
  var candidates = coll.filterDate(d0.advance(-16, 'day'), d1);
  var weighted = candidates.map(function(img) {
    var interval = periodFn(img);
    var overlap = overlapDays(interval, d0, d1);
    var length = ee.Number(interval.get(1)).subtract(interval.get(0)).divide(86400000);
    var value = preprocess(img);
    var rate = ee.Image(ee.Algorithms.If(isTotal, value.divide(length), value));
    var valid = value.mask();
    var coverage = ee.Image.constant(overlap).rename('coverage').updateMask(valid);
    return rate.multiply(overlap).rename('weighted').addBands(coverage).toDouble();
  });
  var sum = weighted.sum()
    .addBands(ee.Image(0).rename('weighted').selfMask(), null, false)
    .addBands(ee.Image(0).rename('coverage').selfMask(), null, false);
  var totalDays = d1.difference(d0, 'day');
  var days = sum.select('coverage');
  var value = ee.Image(ee.Algorithms.If(isMean,
    sum.select('weighted').divide(days.add(0.000001)), sum.select('weighted')));
  return value.updateMask(days.gt(0)).rename(name)
    .addBands(days.divide(totalDays).rename(name + '_coverage'));
}
function landsatCollection(id, red, nir, d0, d1) {
  return ee.ImageCollection(id).filterDate(d0, d1).filterBounds(wells.geometry()).map(function(img) {
    var mask = img.select('QA_PIXEL').bitwiseAnd(63).eq(0).and(img.select('QA_RADSAT').eq(0));
    var r = img.select(red).multiply(0.0000275).add(-0.2);
    var n = img.select(nir).multiply(0.0000275).add(-0.2);
    var denominator = n.add(r);
    return n.subtract(r).divide(denominator).rename('NDVI_Landsat')
      .updateMask(mask.and(denominator.gt(0))).toDouble()
      .copyProperties(img, ['system:time_start']);
  });
}
function monthly(offset) {
  var d0 = start.advance(offset, 'month');
  var d1 = d0.advance(1, 'month');
  var et = weightedComposite(mod16, d0, d1, function(img) { return modisPeriod(img, 8); },
    function(img) {
      var q = img.select('ET_QC');
      var raw = img.select('ET');
      var valid = q.bitwiseAnd(1).eq(0).and(q.rightShift(5).bitwiseAnd(7).lte(1))
        .and(raw.gte(0)).and(raw.lte(32700));
      return raw.multiply(0.1).updateMask(valid);
    }, true, false, 'ET_MODIS_mm');
  var aet = weightedComposite(wapor, d0, d1, dekadPeriod,
    function(img) { return img.select('L1_AETI_D').multiply(0.1); },
    false, false, 'ET_WaPOR_mm');
  var lst = weightedComposite(mod11, d0, d1, function(img) { return modisPeriod(img, 8); },
    function(img) {
      var valid = img.select('QC_Day').bitwiseAnd(3).eq(0);
      return img.select('LST_Day_1km').multiply(0.02).subtract(273.15).updateMask(valid);
    }, false, true, 'LST_C');
  var ndvi = weightedComposite(mod13, d0, d1, function(img) { return modisPeriod(img, 16); },
    function(img) { return img.select('NDVI').multiply(0.0001).updateMask(img.select('SummaryQA').eq(0)); },
    false, true, 'NDVI_MODIS');
  var land = landsatCollection('LANDSAT/LT05/C02/T1_L2', 'SR_B3', 'SR_B4', d0, d1)
    .merge(landsatCollection('LANDSAT/LE07/C02/T1_L2', 'SR_B3', 'SR_B4', d0, d1))
    .merge(landsatCollection('LANDSAT/LC08/C02/T1_L2', 'SR_B4', 'SR_B5', d0, d1))
    .merge(landsatCollection('LANDSAT/LC09/C02/T1_L2', 'SR_B4', 'SR_B5', d0, d1));
  var landMean = safeMean(land, 'NDVI_Landsat', 'NDVI_Landsat');
  var era = era5.filterDate(d0, d1).first();
  var eraFallback = ee.Image.constant([0, 0, 0, 0, 0])
    .rename(['runoff_sum', 'sub_surface_runoff_sum', 'snowfall_sum', 'snowmelt_sum', 'total_evaporation_sum'])
    .selfMask();
  var eraImg = ee.Image(ee.Algorithms.If(era, era, eraFallback));
  var precip = chirps.filterDate(d0, d1).select('precipitation').sum()
    .addBands(ee.Image(0).rename('precipitation').selfMask(), null, false)
    .select('precipitation').rename('precipitation_mm');
  var temp = tc.filterDate(d0, d1);
  return precip
    .addBands(et).addBands(aet).addBands(lst).addBands(ndvi).addBands(landMean)
    .addBands(land.count().addBands(ee.Image(0)).select([0], ['Landsat_scene_count']).toDouble())
    .addBands(eraImg.select('runoff_sum').multiply(1000).rename('Runoff_Total_mm'))
    .addBands(eraImg.select('sub_surface_runoff_sum').multiply(1000).rename('Runoff_SubSurface_mm'))
    .addBands(eraImg.select('snowfall_sum').multiply(1000).rename('Snowfall_mm'))
    .addBands(eraImg.select('snowmelt_sum').multiply(1000).rename('Snowmelt_mm'))
    .addBands(eraImg.select('total_evaporation_sum').multiply(-1000).rename('TotalEvaporation_mm'))
    .addBands(safeMean(temp, 'pdsi', 'PDSI').multiply(0.01))
    .addBands(safeMean(temp, 'def', 'DEF_mm').multiply(0.1))
    .addBands(safeMean(gldas.filterDate(d0, d1), 'SoilMoi0_10cm_inst', 'SoilMoisture_kgm2'))
    .addBands(safeMean(gldas.filterDate(d0, d1), 'SoilTMP0_10cm_inst', 'SoilTemp_K'))
    .addBands(terrain).toDouble().set('system:time_start', d0.millis());
}
var monthlyCollection = ee.ImageCollection(months.map(monthly));
var series = monthlyCollection.map(function(img) {
  var d = ee.Date(img.get('system:time_start')).format('YYYY-MM');
  return img.reduceRegions({collection: wells, reducer: ee.Reducer.mean(), scale: 500, tileScale: 4})
    .map(function(f) { return f.set('date', d).set('export_protocol', 'corrected_v1'); });
}).flatten();
print('Preview:', series.limit(10));
Export.table.toDrive({collection: series, description: 'Haouz_features_corrected_v1',
  folder: 'earthengine', fileFormat: 'CSV'});
